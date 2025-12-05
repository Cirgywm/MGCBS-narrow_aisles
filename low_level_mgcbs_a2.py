from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time as ti
import heapq
import math

from types_common import GridPos, Time
from grid import GridMap
from agent import AgentTask
from constraints import ConstraintTable
from intervals import (
    SafeInterval,
    GoalSafeInterval,
    build_safe_intervals_for_agent,
    build_goal_safe_intervals,
)
from heuristics import mst_heuristic
from low_level_astar import single_agent_astar_to_interval
from metrics import SearchMetrics


@dataclass
class LowLevelState:
    """State A* di low-level MGCBS (DSS)."""
    visited_mask: int
    current_gsi: GoalSafeInterval
    time: Time
    g: float
    h: float
    parent: Optional["LowLevelState"] = None

    def f(self) -> float:
        return self.g + self.h

    def __lt__(self, other: "LowLevelState") -> bool:
        return self.f() < other.f()


class MGCBSLowLevelA2:
    """Low-level solver MGCBS A2 untuk satu agent."""

    def __init__(
        self,
        grid: GridMap,
        agent: AgentTask,
        dist_table: Dict[GridPos, Dict[GridPos, int]],
        time_horizon: Time,
        metrics: Optional[SearchMetrics] = None,
        deadline: Optional[float] = None,
    ) -> None:
        self.grid = grid
        self.agent = agent
        self.dist_table = dist_table
        self.time_horizon = time_horizon
        self.metrics = metrics
        self.deadline = deadline

    def plan_multi_goal_path(
        self,
        constraints: ConstraintTable,
    ) -> Optional[List[GridPos]]:
        """Hitung jalur multi-goal untuk agent ini di bawah constraint."""
        si_by_vertex = build_safe_intervals_for_agent(
            self.grid,
            self.agent,
            constraints,
            self.time_horizon,
        )
        gsi_by_goal = build_goal_safe_intervals(self.agent, si_by_vertex)

        # SI awal di start pada t=0
        start_si_list = si_by_vertex.get(self.agent.start, [])
        start_si = None
        for si in start_si_list:
            if si.contains(0):
                start_si = si
                break
        if start_si is None:
            return None

        start_gsi = GoalSafeInterval(goal_index=-1, interval=start_si)

        num_goals = len(self.agent.task_goals)
        all_visited_mask = (1 << num_goals) - 1

        start_state = LowLevelState(
            visited_mask=0,
            current_gsi=start_gsi,
            time=0,
            g=0.0,
            h=mst_heuristic(
                current_pos=self.agent.start,
                unvisited_goals=self.agent.task_goals,
                dist_table=self.dist_table,
            ),
            parent=None,
        )

        open_heap: List[Tuple[float, LowLevelState]] = []
        heapq.heappush(open_heap, (start_state.f(), start_state))

        visited: Dict[Tuple[int, GridPos, Time], float] = {
            (0, self.agent.start, 0): 0.0
        }

        while open_heap:
            # cek timeout
            if self.deadline is not None and ti.perf_counter() > self.deadline:
                return None  # berhenti karena timeout
            
            _, cur = heapq.heappop(open_heap)
            
            # hitung state DSS yang di-expand
            if self.metrics is not None:
                self.metrics.low_level_states_expanded += 1

            if cur.visited_mask == all_visited_mask:
                return self._reconstruct_full_path(cur, constraints)

            key = (cur.visited_mask, cur.current_gsi.vertex, cur.time)
            if cur.g > visited.get(key, math.inf):
                continue

            unvisited_indices = [
                idx for idx in range(num_goals) if not (cur.visited_mask & (1 << idx))
            ]

            if not unvisited_indices:
                continue

            for idx in unvisited_indices:
                gsi_list = gsi_by_goal.get(idx, [])
                if not gsi_list:
                    continue

                # last-goal rule
                if len(unvisited_indices) == 1:
                    candidates = [gsi_list[-1]]
                else:
                    candidates = gsi_list

                for gsi in candidates:
                    res = single_agent_astar_to_interval(
                        grid=self.grid,
                        agent=self.agent,
                        constraints=constraints,
                        start_pos=cur.current_gsi.vertex,
                        start_time=cur.time,
                        target=gsi,
                        time_horizon=self.time_horizon,
                        metrics=self.metrics,
                        deadline=self.deadline,
                    )
                    if res is None:
                        continue
                    cost_len, _ = res
                    arrival_time = cur.time + cost_len

                    new_mask = cur.visited_mask | (1 << idx)
                    new_g = arrival_time
                    new_pos = gsi.vertex
                    remaining_goals = [
                        self.agent.task_goals[j]
                        for j in range(num_goals)
                        if not (new_mask & (1 << j))
                    ]
                    new_h = mst_heuristic(
                        current_pos=new_pos,
                        unvisited_goals=remaining_goals,
                        dist_table=self.dist_table,
                    )
                    new_state = LowLevelState(
                        visited_mask=new_mask,
                        current_gsi=gsi,
                        time=arrival_time,
                        g=float(new_g),
                        h=float(new_h),
                        parent=cur,
                    )

                    new_key = (new_mask, new_pos, arrival_time)
                    if new_g < visited.get(new_key, math.inf):
                        visited[new_key] = new_g
                        heapq.heappush(open_heap, (new_state.f(), new_state))

        return None
    
    def make_GSI_for_loading(self, pos):
        si = SafeInterval(pos, 0, self.time_horizon)
        return GoalSafeInterval(-99, si)

    def _reconstruct_full_path(
        self,
        final_state: LowLevelState,
        constraints: ConstraintTable,
    ) -> List[GridPos]:
        """Bangun path posisi per time-step dari chain state GSI."""
        seq: List[LowLevelState] = []
        cur = final_state
        while cur is not None:
            seq.append(cur)
            cur = cur.parent
        seq.reverse()

        full_path: List[GridPos] = []

        # Bangun path dari DSS (GOALS RAK)
        for i in range(len(seq) - 1):
            s_prev = seq[i]
            s_next = seq[i + 1]

            start_pos = s_prev.current_gsi.vertex
            start_time = s_prev.time
            gsi = s_next.current_gsi

            res = single_agent_astar_to_interval(
                grid=self.grid,
                agent=self.agent,
                constraints=constraints,
                start_pos=start_pos,
                start_time=start_time,
                target=gsi,
                time_horizon=self.time_horizon,
                metrics=self.metrics,
                deadline=self.deadline,
            )
            if res is None:
                # raise RuntimeError(
                #     "Seharusnya ada path di sini (konsistensi low-level A*)."
                # )
                return None

            _, subpath = res  # (pos, time)
            if i == 0:
                full_path.extend([p for p, t in subpath])
            else:
                full_path.extend([p for p, t in subpath[1:]])
                
        # => Loading goal (bukan bagian DSS)
        last_pos = full_path[-1]
        last_t = len(full_path) - 1

        loading = self.agent.loading_goal
        loading_gsi = self.make_GSI_for_loading(loading)

        res = single_agent_astar_to_interval(
            grid=self.grid,
            agent=self.agent,
            constraints=constraints,
            start_pos=last_pos,
            start_time=last_t,
            target=loading_gsi,
            time_horizon=self.time_horizon,
            metrics=self.metrics,
            deadline=self.deadline,
        )
        if res is None:
            return None

        _, subpath = res
        full_path.extend([p for p, t in subpath[1:]])

        return full_path
