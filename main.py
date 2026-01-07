from __future__ import annotations

from typing import List, Tuple, Optional
import os
import random
import time
import csv
import json
from multiprocessing import Process, Queue

from grid import GridMap
from agent import AgentTask
from heuristics import manhattan
from high_level_mgcbs_a2 import MGCBSSolverA2
from types_common import GridPos
from metrics import SearchMetrics
from visualization import visualize_paths_per_timestep


# ----------------------------------------------------------------------
# 1. Fungsi untuk membangun layout warehouse
# ----------------------------------------------------------------------
def create_warehouse_grid() -> GridMap:
    width, height = 38, 24
    grid = GridMap(width, height)

    # 1) border walls
    walls: List[GridPos] = []
    for x in range(width):
        walls.append((x, 0))
        walls.append((x, height - 1))
    for y in range(height):
        walls.append((0, y))
        walls.append((width - 1, y))

    # 2) shelves + narrow aisles + intersections
    shelves: List[GridPos] = []
    narrow_aisles: List[GridPos] = []
    intersections: List[GridPos] = []

    # parameter posisi rak (silakan sesuaikan dengan desainmu)
    shelf_start_x = 10
    shelf_width = 8
    shelf_gap = 1

    # 3 kolom blok rak, tiap blok berisi 7 baris rak
    for column in range(3):
        x_start = shelf_start_x + column * (shelf_width + shelf_gap)
        for row in range(7):
            y_start = 2 + row * 3
            # rak 2 sel tinggi (y dan y+1)
            for x in range(x_start, x_start + shelf_width):
                for y in range(y_start, y_start + 1):
                    shelves.append((x, y))
                    shelves.append((x, y + 1))

                    # narrow aisle di depan rak (y-1) dan belakang rak (untuk baris terakhir)
                    narrow_aisles.append((x, y - 1))
                    if row == 6:
                        narrow_aisles.append((x, y + 2))

                    # intersection di kanan blok rak
                    if x - x_start == shelf_width - 1:
                        narrow_aisles.append((x + 1, y))
                        narrow_aisles.append((x + 1, y + 1))
                        intersections.append((x + 1, y - 1))
                        if row == 6:
                            intersections.append((x + 1, y + 2))

    # 3) loading stations (hijau)
    stations: List[GridPos] = [
        (1, 4),
        (1, 6),
        (1, 8),
        (1, 10),
        (1, 13),
        (1, 15),
        (1, 17),
        (1, 19),
    ]

    # 4) docking stations (kuning)
    dockings: List[GridPos] = [
        (1, 1),
        (3, 1),
        (5, 1),
        (7, 1),
        (1, 22),
        (3, 22),
        (5, 22),
        (7, 22),
    ]

    grid.add_walls(walls)
    grid.add_shelves(shelves)
    grid.add_stations(stations)
    grid.add_dockings(dockings)
    grid.add_narrowaisles(narrow_aisles)
    grid.add_intersections(intersections)

    return grid


# ----------------------------------------------------------------------
# 2. Helper: akses vertikal terdekat untuk sebuah shelf
# ----------------------------------------------------------------------
def get_vertical_access_cell(grid: GridMap, shelf: GridPos) -> Optional[GridPos]:
    """Cari sel kosong vertikal dari sebuah rak.
    Prioritas: sel atas (x, y-1), kalau tidak bisa pakai bawah (x, y+1).
    Return None jika keduanya tidak bisa dilalui.
    """
    x, y = shelf
    candidates = [(x, y - 1), (x, y + 1)]

    valid: List[GridPos] = []
    for c in candidates:
        if grid.in_bounds(c) and grid.passable(c):
            valid.append(c)

    if not valid:
        return None

    # Prioritaskan sel atas jika tersedia (sesuai contoh (10,2) -> (10,1))
    if candidates[0] in valid:
        return candidates[0]
    return valid[0]


# ----------------------------------------------------------------------
# 3. Generate agents & tasks untuk satu skenario
# ----------------------------------------------------------------------
def generate_agents_for_scenario(
    grid: GridMap,
    num_agents: int,
    tasks_per_agent: int,
    seed: int = 0,
) -> List[AgentTask]:
    """Bangun daftar AgentTask untuk satu kombinasi (num_agents, tasks_per_agent)."""

    rng = random.Random(seed)

    dockings = grid.get_docking_stations()
    loadings = grid.get_stations()
    shelves_all = grid.get_shelves()

    assert len(dockings) >= num_agents, "Docking station tidak cukup untuk jumlah agen."
    assert len(loadings) >= num_agents, "Loading station tidak cukup untuk jumlah agen."

    # pilih docking & loading secara acak tapi unik
    docking_chosen = rng.sample(dockings, num_agents)
    loading_chosen = rng.sample(loadings, num_agents)

    # pilih rak yang punya akses vertikal
    candidate_shelves: List[GridPos] = []
    for s in shelves_all:
        access = get_vertical_access_cell(grid, s)
        if access is not None:
            candidate_shelves.append(s)

    total_tasks = num_agents * tasks_per_agent
    assert len(candidate_shelves) >= total_tasks, "Rak dengan akses vertikal tidak cukup."

    # shelves_chosen = rng.sample(candidate_shelves, total_tasks)
    # shelves_chosen = [rng.choice(candidate_shelves) for _ in range(total_tasks)]
    # pilih rak untuk tiap agen secara acak tapi unik dalam agen yang sama
    shelves_for_agents: List[List[GridPos]] = []
    
    for aid in range(num_agents):
            chosen: List[GridPos] = []
            chosen_set = set()
            while len(chosen) < tasks_per_agent:
                s = rng.choice(candidate_shelves)  # with replacement global
                if s in chosen_set:
                    continue  # hindari duplikat dalam agen yang sama
                chosen_set.add(s)
                chosen.append(s)
            shelves_for_agents.append(chosen)

    agents: List[AgentTask] = []
    # idx = 0
    for aid in range(num_agents):
        start = docking_chosen[aid]
        loading_goal = loading_chosen[aid]

        # rak untuk agen ini
        # shelves_for_agent = shelves_chosen[idx:idx + tasks_per_agent]
        # idx += tasks_per_agent
        shelves_for_agent = shelves_for_agents[aid]

        goals: List[GridPos] = []
            
        for sh in shelves_for_agent:
            access_cell = get_vertical_access_cell(grid, sh)
            if access_cell is None:
                # safety
                continue
            goals.append(access_cell)

        agents.append(
            AgentTask(
                agent_id=aid, 
                start=start, 
                task_goals=goals, 
                shelves=shelves_for_agent, 
                loading_goal=loading_goal
                )
            )

    return agents


# ----------------------------------------------------------------------
# 4. Estimasi time horizon
# ----------------------------------------------------------------------
# def estimate_time_horizon(grid: GridMap, tasks_per_agent: int) -> int:
#     """Estimasi sederhana batas waktu per agent.

#     Bisa di tweak: makin besar kalau sering 'no solution karena horizon'.
#     """
#     return (grid.width + grid.height) * (tasks_per_agent + 2)

def estimate_time_horizon(grid: GridMap, agents: List[AgentTask]) -> int:
    """Estimasi horizon berdasarkan total jarak Manhattan path ideal."""
    max_est = 0
    for ag in agents:
        est = manhattan(ag.start, ag.task_goals[0])
        
        for i in range(len(ag.task_goals) - 1):
            est += manhattan(ag.task_goals[i], ag.task_goals[i + 1])
            
        if ag.loading_goal is not None:
            est += manhattan(ag.task_goals[-1], ag.loading_goal)
            
        if est > max_est:
            max_est = est
    # Faktor pengaman (mis. 2x atau 3x) + padding
    return int(max_est * 2) + 10


# ----------------------------------------------------------------------
# 5. Jalankan satu skenario
# ----------------------------------------------------------------------
def run_single_scenario(
    num_agents: int,
    tasks_per_agent: int,
    seed: int = 0,
    iteration: int = 1,
    verbose: bool = True,
    time_limit: float = 30.0,
):
    all_summary = []
    all_detailed = []

    grid = create_warehouse_grid()
    agents = generate_agents_for_scenario(
        grid=grid,
        num_agents=num_agents,
        tasks_per_agent=tasks_per_agent,
        seed=seed,
    )

    time_horizon = estimate_time_horizon(grid, agents)

    solver = MGCBSSolverA2(
        grid=grid,
        agents=agents,
        time_horizon=time_horizon,
    )

    start = time.perf_counter()
    deadline = start + time_limit

    solution = solver.solve(deadline=deadline)
    end = time.perf_counter()
    elapsed = end - start

    status = "success"
    notes = ""

    if solution is None:
        if solver.timed_out:
            status = "timeout"
            notes = "solver exceeded time limit"
        else:
            status = "fail"
            notes = "no feasible solution in given horizon"

        # tetap buat summary row dalam kasus fail / timeout
        summary_row = {
            "iteration": iteration,
            "status": status,
            "num_agents": num_agents,
            "tasks_per_agent": tasks_per_agent,
            "seed": seed,
            "horizon": time_horizon,
            "elapsed_ms": elapsed * 1000.0,
            "cbs_nodes": solver.metrics.cbs_nodes_expanded,
            "lowlevel_states": solver.metrics.low_level_states_expanded,
            "astar_states": solver.metrics.low_level_astar_expanded,
            "conflicts_found": solver.metrics.conflicts_found,
            "conflicts_resolved": solver.metrics.conflicts_resolved,
            "notes": notes,
        }

        # detail agent tetap dicatat (tanpa path)
        detail = {
            "iteration": iteration,
            "status": status,
            "num_agents": num_agents,
            "tasks_per_agent": tasks_per_agent,
            "seed": seed,
            "horizon": time_horizon,
            "elapsed_ms": elapsed * 1000.0,
            "notes": notes,
            "metrics": {
                "cbs_nodes": solver.metrics.cbs_nodes_expanded,
                "lowlevel_states": solver.metrics.low_level_states_expanded,
                "astar_states": solver.metrics.low_level_astar_expanded,
                "conflicts_found": solver.metrics.conflicts_found,
                "conflicts_resolved": solver.metrics.conflicts_resolved,
            },
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "start": list(a.start),
                    "task_goals": [list(g) for g in a.task_goals],
                    "shelves": [list(s) for s in a.shelves],
                    "loading_goal": list(a.loading_goal) if a.loading_goal else None,
                }
                for a in agents
            ],
            "solution_paths": None,
        }

        all_summary.append(summary_row)
        all_detailed.append(detail)

        if verbose:
            print(f"[{status.upper()}] Agen={num_agents}, task={tasks_per_agent}, seed={seed}")
            print(f"  Notes: {notes}")
            print(f"  Metrics: CBS={solver.metrics.cbs_nodes_expanded}, "
                  f"DSS={solver.metrics.low_level_states_expanded}, "
                  f"A*={solver.metrics.low_level_astar_expanded}")

    else:
        # SUCCESS CASE -----------------------

        soc = sum(len(p) - 1 for p in solution.values())
        makespan = max(len(p) - 1 for p in solution.values())

        summary_row = {
            "iteration": iteration,
            "status": "success",
            "num_agents": num_agents,
            "tasks_per_agent": tasks_per_agent,
            "seed": seed,
            "horizon": time_horizon,
            "soc": soc,
            "makespan": makespan,
            "elapsed_ms": elapsed * 1000.0,
            "cbs_nodes": solver.metrics.cbs_nodes_expanded,
            "lowlevel_states": solver.metrics.low_level_states_expanded,
            "astar_states": solver.metrics.low_level_astar_expanded,
            "conflicts_found": solver.metrics.conflicts_found,
            "conflicts_resolved": solver.metrics.conflicts_resolved,
        }

        detail = {
            "iteration": iteration,
            "status": "success",
            "num_agents": num_agents,
            "tasks_per_agent": tasks_per_agent,
            "seed": seed,
            "horizon": time_horizon,
            "metrics": summary_row,
            "agents": [
                {
                    "agent_id": a.agent_id,
                    "start": list(a.start),
                    "task_goals": [list(g) for g in a.task_goals],
                    "shelves": [list(s) for s in a.shelves],
                    "loading_goal": list(a.loading_goal) if a.loading_goal else None,
                    "path": [list(p) for p in solution[a.agent_id]],
                }
                for a in agents
            ]
        }

        all_summary.append(summary_row)
        all_detailed.append(detail)

        if verbose:
            print(f"[OK] Agen={num_agents}, task={tasks_per_agent}, seed={seed}")
            
        # if status == "success":
        #     # Visualize solution
        #     scenario_name = f"na{num_agents}_nt{tasks_per_agent}_seed{seed}"
        #     out_dir = os.path.join("figures", f"iter{iteration}", scenario_name)

        #     visualize_paths_per_timestep(
        #         grid=grid,
        #         agents=agents,
        #         solution=solution,
        #         out_dir=out_dir,
        #         scenario_name=scenario_name,
        #         draw_trails=True,
        #     )

    # ----- simpan ke CSV -----
    if all_summary:
        # Union dari semua fieldnames
        fieldnames = set()
        for row in all_summary:
            for k in row.keys():
                fieldnames.add(k)

        # Convert ke list
        fieldnames = list(sorted(fieldnames))

        with open("mgcbs_results_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in all_summary:
                safe_row = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(safe_row)

        print("\nSummary disimpan ke mgcbs_results_summary.csv")

    # ----- JSON detail -----
    with open("mgcbs_results_detailed.json", "w") as f:
        json.dump(all_detailed, f, indent=2)
    print("Detail disimpan ke mgcbs_results_detailed.json")

    return summary_row, detail


# ----------------------------------------------------------------------
# 6. Jalankan kombinasi eksperimen
# ----------------------------------------------------------------------
def run_all_experiments(iterations: int = 3):
    agent_variants = [2, 4, 5, 6, 8]
    tasks_variants = [1, 2, 3, 5]
    iteration_variants = iterations

    all_summary = []
    all_detailed = []

    print("=== Menjalankan eksperimen MGCBS A2 di warehouse ===")
    for it in range(1, iteration_variants + 1):
        print(f"\n=== Iterasi {it} ===")
        for na in agent_variants:
            for nt in tasks_variants:
                # seed
                seed = it * 1000 + 10 * na + nt
                print("\n----------------------------------------")
                print(f"Skenario: {na} agen, {nt} task per agen")
                
                result = run_single_scenario(
                    num_agents=na,
                    tasks_per_agent=nt,
                    seed=seed,
                    iteration=it,
                    verbose=True,
                    time_limit=120.0,
                )
                
                # if result is None:
                #     # simpan entry gagal
                #     summary_row = {
                #         "num_agents": na,
                #         "tasks_per_agent": nt,
                #         "seed": seed,
                #         "status": "fail_or_timeout"
                #     }
                #     all_summary.append(summary_row)
                #     # detail tidak ada → skip
                #     continue
                
                summary_row, detailed_record = result
                summary_row["status"] = "success"
                all_summary.append(summary_row)
                all_detailed.append(detailed_record)

    # ----- simpan ke CSV -----
    if all_summary:
        # Union dari semua fieldnames
        fieldnames = set()
        for row in all_summary:
            for k in row.keys():
                fieldnames.add(k)

        # Convert ke list
        fieldnames = list(sorted(fieldnames))

        with open("mgcbs_results_summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in all_summary:
                safe_row = {key: row.get(key, "") for key in fieldnames}
                writer.writerow(safe_row)

        print("\nSummary disimpan ke mgcbs_results_summary.csv")

    # ----- JSON detail -----
    with open("mgcbs_results_detailed.json", "w") as f:
        json.dump(all_detailed, f, indent=2)
    print("Detail disimpan ke mgcbs_results_detailed.json")


if __name__ == "__main__":
    run_single_scenario(
        num_agents=2,
        tasks_per_agent=2,
        seed=1022,
        verbose=True,
        time_limit=120,
    )
    
    # run_all_experiments(iterations=3)
    
    # grid = create_warehouse_grid()
    # grid.visualize()
