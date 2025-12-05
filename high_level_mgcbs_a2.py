from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time as ti
import heapq

from types_common import GridPos, Time
from grid import GridMap
from agent import AgentTask
from constraints import (
    ConstraintTable,
    VertexConstraint,
    EdgeConstraint,
)
from heuristics import compute_distance_table_for_agent
from low_level_mgcbs_a2 import MGCBSLowLevelA2
from metrics import SearchMetrics


@dataclass
class CBSNode:
    """Node di constraint tree CBS untuk MGCBS."""
    constraints: ConstraintTable
    paths: Dict[int, List[GridPos]]
    cost: int

    def __lt__(self, other: "CBSNode") -> bool:
        return self.cost < other.cost


def detect_first_conflict(
    paths: Dict[int, List[GridPos]]
) -> Optional[Tuple[int, int, GridPos, Optional[Tuple[GridPos, GridPos]]]]:
    """Cari konflik paling awal di antara semua pasangan agent."""
    if not paths:
        return None

    agent_ids = list(paths.keys())
    max_t = max(len(p) for p in paths.values()) - 1

    def get_pos(aid: int, t: int) -> GridPos:
        path = paths[aid]
        if t < len(path):
            return path[t]
        return path[-1]

    for t in range(max_t + 1):
        # vertex conflict
        pos_at_t: Dict[GridPos, List[int]] = {}
        for aid in agent_ids:
            pos = get_pos(aid, t)
            pos_at_t.setdefault(pos, []).append(aid)
        for pos, agents_here in pos_at_t.items():
            if len(agents_here) > 1:
                a1, a2 = agents_here[0], agents_here[1]
                return (t, a1, pos, None)

        # edge conflict
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a1 = agent_ids[i]
                a2 = agent_ids[j]
                p1 = get_pos(a1, t)
                p1_next = get_pos(a1, t + 1)
                p2 = get_pos(a2, t)
                p2_next = get_pos(a2, t + 1)
                if p1 == p2_next and p2 == p1_next and p1 != p2:
                    return (t + 1, a1, p1, (p1, p1_next))

    return None


class MGCBSSolverA2:
    """High-level MGCBS A2 (tanpa TIS Forest)."""

    def __init__(
        self,
        grid: GridMap,
        agents: List[AgentTask],
        time_horizon: Time,
    ) -> None:
        self.grid = grid
        self.agents = agents
        self.time_horizon = time_horizon
        self.dist_tables: Dict[int, Dict[GridPos, Dict[GridPos, int]]] = {
            a.agent_id: compute_distance_table_for_agent(grid, a) for a in agents
        }
        self.metrics = SearchMetrics()
        self.timed_out: bool = False       # <--- FLAG

    def solve(self, deadline: Optional[float] = None) -> Optional[Dict[int, List[GridPos]]]:
        """Jalankan MGCBS A2."""
        # reset metrics
        self.metrics.cbs_nodes_expanded = 0
        self.metrics.low_level_states_expanded = 0
        self.metrics.low_level_astar_expanded = 0
        self.metrics.conflicts_found = 0
        self.metrics.conflicts_resolved = 0
        self.timed_out = False
        
        root_constraints = ConstraintTable()
        root_paths: Dict[int, List[GridPos]] = {}

        for a in self.agents:
            low = MGCBSLowLevelA2(
                grid=self.grid,
                agent=a,
                dist_table=self.dist_tables[a.agent_id],
                time_horizon=self.time_horizon,
                metrics=self.metrics,
                deadline=deadline,
            )
            path = low.plan_multi_goal_path(root_constraints)
            if path is None:
                # bisa karena buntu atau timeout, tapi kalau timeout
                # di low-level, kita treat sebagai timeout juga
                if deadline is not None and ti.perf_counter() > deadline:
                    self.timed_out = True
                return None
            root_paths[a.agent_id] = path

        root_cost = sum(len(p) - 1 for p in root_paths.values())
        root = CBSNode(constraints=root_constraints, paths=root_paths, cost=root_cost)

        open_heap: List[Tuple[int, CBSNode]] = []
        heapq.heappush(open_heap, (root.cost, root))

        while open_heap:
            # cek timeout di high-level
            if deadline is not None and ti.perf_counter() > deadline:
                self.timed_out = True
                return None
            
            _, node = heapq.heappop(open_heap)
            
            # node CBS yang di-expand
            self.metrics.cbs_nodes_expanded += 1

            conflict = detect_first_conflict(node.paths)
            if conflict is None:
                return node.paths
            
            # konflik ditemukan
            self.metrics.conflicts_found += 1

            time, a1, pos, edge = conflict

            # cari agent lain yg konflik
            other_agent_id = None
            if edge is None:
                # vertex conflict
                for aid, path in node.paths.items():
                    if aid == a1:
                        continue
                    p_at_t = path[time] if time < len(path) else path[-1]
                    if p_at_t == pos:
                        other_agent_id = aid
                        break
            else:
                # edge conflict
                for aid, path in node.paths.items():
                    if aid == a1:
                        continue
                    p_prev = path[time - 1] if time - 1 < len(path) else path[-1]
                    p_now = path[time] if time < len(path) else path[-1]
                    if p_prev == edge[1] and p_now == edge[0]:
                        other_agent_id = aid
                        break

            if other_agent_id is None:
                continue

            children: List[CBSNode] = []
            child_created = False   # <--- FLAG

            for agent_to_constrain in [a1, other_agent_id]:
                # copy constraint lama
                new_constraints = ConstraintTable(
                    vertex_constraints={
                        aid: dict(vc)
                        for aid, vc in node.constraints.vertex_constraints.items()
                    },
                    edge_constraints={
                        aid: dict(ec)
                        for aid, ec in node.constraints.edge_constraints.items()
                    },
                )

                # tambah constraint baru
                if edge is None:
                    new_constraints.add_vertex_constraint(
                        VertexConstraint(agent_id=agent_to_constrain, pos=pos, time=time)
                    )
                else:
                    path = node.paths[agent_to_constrain]
                    from_pos = path[time - 1] if time - 1 < len(path) else path[-1]
                    to_pos = path[time] if time < len(path) else path[-1]
                    new_constraints.add_edge_constraint(
                        EdgeConstraint(
                            agent_id=agent_to_constrain,
                            from_pos=from_pos,
                            to_pos=to_pos,
                            time=time,
                        )
                    )

                new_paths = {aid: list(p) for aid, p in node.paths.items()}

                agent_obj = next(a for a in self.agents if a.agent_id == agent_to_constrain)
                low = MGCBSLowLevelA2(
                    grid=self.grid,
                    agent=agent_obj,
                    dist_table=self.dist_tables[agent_to_constrain],
                    time_horizon=self.time_horizon,
                    metrics=self.metrics,
                    deadline=deadline,
                )
                new_path = low.plan_multi_goal_path(new_constraints)
                if new_path is None:
                    # bisa buntu atau timeout
                    if deadline is not None and ti.perf_counter() > deadline:
                        self.timed_out = True
                        return None
                    continue  # child invalid, skip

                new_paths[agent_to_constrain] = new_path
                new_cost = sum(len(p) - 1 for p in new_paths.values())

                children.append(
                    CBSNode(
                        constraints=new_constraints,
                        paths=new_paths,
                        cost=new_cost,
                    )
                )
                child_created = True   # <--- set jika child valid

            if child_created:
                self.metrics.conflicts_resolved += 1

            for ch in children:
                heapq.heappush(open_heap, (ch.cost, ch))

        return None
