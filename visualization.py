import os
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from types_common import GridPos
from grid import GridMap
from agent import AgentTask


def _draw_static_map(ax, grid: GridMap):
    """Gambar elemen-elemen statis warehouse (walls, shelves, docking, dll)."""
    # Ambil info dari GridMap
    walls = grid.get_walls()               # cell dengan kode 1
    shelves = grid.get_shelves()           # kode 2
    stations = grid.get_stations()         # kode 3 (loading)
    docks = grid.get_docking_stations()    # kode 4 (docking)
    narrow = grid.get_narrowaisles() if hasattr(grid, "get_narrowaisles") else []
    inters = grid.get_intersections() if hasattr(grid, "get_intersections") else []

    # 1. Wall / obstacle (hitam)
    for (x, y) in walls:
        ax.add_patch(
            Rectangle((x, y), 1, 1,
                      facecolor="black", edgecolor="black", linewidth=0.5)
        )

    # 2. Shelves (rak, biru/coklat sesuai selera)
    for (x, y) in shelves:
        ax.add_patch(
            Rectangle((x, y), 1, 1,
                      facecolor="blue", edgecolor="black", linewidth=0.5)
        )

    # 3. Narrow aisles (papayawhip seperti di grid.visualize)
    for (x, y) in narrow:
        ax.add_patch(
            Rectangle((x, y), 1, 1,
                      facecolor="papayawhip", edgecolor="lightgray", linewidth=0.3)
        )

    # 4. Intersections (abu muda)
    for (x, y) in inters:
        ax.add_patch(
            Rectangle((x, y), 1, 1,
                      facecolor="lightgray", edgecolor="lightgray", linewidth=0.3)
        )

    # 5. Loading stations (hijau)
    for (x, y) in stations:
        ax.add_patch(
            Rectangle((x, y), 1, 1,
                      facecolor="green", edgecolor="black", linewidth=0.8)
        )

    # 6. Docking stations (kuning)
    for (x, y) in docks:
        ax.add_patch(
            Rectangle((x, y), 1, 1,
                      facecolor="yellow", edgecolor="black", linewidth=0.8)
        )


def _compute_task_order_for_agent(
    path: List[GridPos],
    task_goals: List[GridPos],
) -> List[tuple[GridPos, int]]:
    """Kembalikan list (goal_pos, order_index) berdasarkan waktu kunjungan pertama.

    order_index = 1, 2, 3, ... sesuai urutan kunjungan.
    """
    first_visit_time = []

    for g in task_goals:
        t_hit = None
        for t, pos in enumerate(path):
            if pos == g:
                t_hit = t
                break
        if t_hit is not None:
            first_visit_time.append((g, t_hit))

    # sort by waktu kunjungan
    first_visit_time.sort(key=lambda x: x[1])

    # beri nomor urut 1..k
    ordered = []
    for order_idx, (g, _) in enumerate(first_visit_time, start=1):
        ordered.append((g, order_idx))
    return ordered


def visualize_paths_per_timestep(
    grid: GridMap,
    agents: List[AgentTask],
    solution: Dict[int, List[GridPos]],
    out_dir: str,
    scenario_name: str = "scenario",
    draw_trails: bool = True,
):
    """Gambar visualisasi jalur tiap agen di tiap timestep dan simpan ke file gambar.

    Parameters
    ----------
    grid : GridMap
        Warehouse layout.
    agents : List[AgentTask]
        Daftar agent (start, shelves, loading).
    solution : Dict[int, List[GridPos]]
        Path per agent (list posisi tiap timestep).
    out_dir : str
        Folder output untuk menyimpan gambar.
    scenario_name : str
        Nama skenario (akan dipakai di nama file).
    draw_trails : bool
        Kalau True, gambar jejak jalur agen sampai timestep saat ini.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Cari jumlah timestep maksimum
    max_t = max(len(p) for p in solution.values())

    # Warna untuk agen (kalau agen > jumlah warna, akan diulang)
    agent_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    for t in range(max_t):
        fig, ax = plt.subplots(figsize=(grid.width / 2.5, grid.height / 2.5))

        # Setting axes
        ax.set_xlim(0, grid.width)
        ax.set_ylim(0, grid.height)
        ax.set_aspect("equal")

        # grid lines
        ax.set_xticks(range(grid.width + 1))
        ax.set_yticks(range(grid.height + 1))
        ax.grid(True, which="both", linewidth=0.3)
        ax.invert_yaxis()  # kalau kamu mau (0,0) di kiri atas, boleh dihapus kalau mau (0,0) di kiri bawah

        # Gambar layout warehouse statis
        _draw_static_map(ax, grid)

        # Gambar agen
        for aid, path in solution.items():
            color = agent_colors[aid % len(agent_colors)]
            
            # Ambil object AgentTask (diasumsikan index list agents = agent_id)
            atask: AgentTask = agents[aid]
            
            task_goals = getattr(atask, "task_goals", getattr(atask, "goals", []))

            # Hitung urutan kunjungan: (goal_pos, order_index)
            ordered_goals = _compute_task_order_for_agent(path, task_goals)

            # Prefix huruf agen: 0->A, 1->B, dst.
            if aid < 26:
                agent_letter = chr(ord("A") + aid)
            else:
                agent_letter = f"A{aid}"  # fallback kalau agent > 26

            # Gambar lingkaran + label (misal A1, A2, ...)
            for (gx, gy), order_idx in ordered_goals:
                label = f"{agent_letter}{order_idx}"
                circ = Circle(
                    (gx + 0.5, gy + 0.5),
                    radius=0.35,
                    edgecolor=color,
                    facecolor="white",
                    linewidth=1.5,
                    alpha=0.9,
                )
                ax.add_patch(circ)
                ax.text(
                    gx + 0.5,
                    gy + 0.5,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

            # posisi agen pada timestep t (kalau path lebih pendek, gunakan posisi terakhir)
            if t < len(path):
                pos = path[t]
            else:
                pos = path[-1]

            x, y = pos

            # Jejak jalur sampai timestep ini
            if draw_trails:
                trail_len = min(t + 1, len(path))
                xs = [p[0] + 0.5 for p in path[:trail_len]]
                ys = [p[1] + 0.5 for p in path[:trail_len]]
                ax.plot(xs, ys, linestyle="-", linewidth=1.0, color=color, alpha=0.7)

            # Agen sebagai kotak berwarna dengan label ID
            ax.add_patch(
                Rectangle(
                    (x, y),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.9,
                )
            )
            ax.text(
                x + 0.5,
                y + 0.5,
                str(aid),
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_title(f"{scenario_name}  |  t = {t}")
        plt.tight_layout()

        # Simpan ke file
        filename = os.path.join(out_dir, f"{scenario_name}_t{t:03d}.png")
        fig.savefig(filename, dpi=150)
        plt.close(fig)
