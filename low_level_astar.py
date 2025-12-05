from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time as ti
import heapq
import math

from types_common import GridPos, Time
from grid import GridMap
from agent import AgentTask
from constraints import ConstraintTable
from intervals import GoalSafeInterval
from heuristics import manhattan
from metrics import SearchMetrics


@dataclass
class AStarNode:
    pos: GridPos
    time: Time
    g: float
    h: float

    def f(self) -> float:
        return self.g + self.h

    def __lt__(self, other: "AStarNode") -> bool:
        return self.f() < other.f()


def single_agent_astar_to_interval(
    grid: GridMap,
    agent: AgentTask,
    constraints: ConstraintTable,
    start_pos: GridPos,
    start_time: Time,
    target: GoalSafeInterval,
    time_horizon: Time,
    metrics: Optional[SearchMetrics] = None,
    deadline: Optional[float] = None,
) -> Optional[Tuple[int, List[Tuple[GridPos, Time]]]]:
    """A* di ruang (x, y, t) untuk mencapai target GSI."""
    goal_pos = target.vertex
    earliest = target.start_t
    latest = target.end_t

    open_heap: List[Tuple[float, AStarNode]] = []

    start_node = AStarNode(
        pos=start_pos,
        time=start_time,
        g=0.0,
        h=manhattan(start_pos, goal_pos),
    )
    heapq.heappush(open_heap, (start_node.f(), start_node))

    visited: Dict[Tuple[GridPos, Time], float] = {(start_pos, start_time): 0.0}
    parent: Dict[Tuple[GridPos, Time], Tuple[GridPos, Time]] = {}
    
    # MAX_ASTAR_EXPANDED = 20000

    while open_heap:
        # cek timeout
        if deadline is not None and ti.perf_counter() > deadline:
            return None  # berhenti karena timeout
        
        _, current = heapq.heappop(open_heap)
        pos, t, g = current.pos, current.time, current.g
        
        # hitung node A* yang di-expand
        if metrics is not None:
            metrics.low_level_astar_expanded += 1
            # if metrics.low_level_astar_expanded > MAX_ASTAR_EXPANDED:
            #     return None

        pos, t, g = current.pos, current.time, current.g

        if t > time_horizon:
            continue

        if pos == goal_pos and earliest <= t <= latest:
            path: List[Tuple[GridPos, Time]] = []
            cur_state = (pos, t)
            while cur_state in parent:
                path.append(cur_state)
                cur_state = parent[cur_state]
            path.append((start_pos, start_time))
            path.reverse()
            return t - start_time, path

        # Wait
        next_t = t + 1
        if next_t <= time_horizon:
            if not constraints.is_vertex_blocked(agent.agent_id, pos, next_t):
                state = (pos, next_t)
                ng = g + 1.0
                if ng < visited.get(state, math.inf):
                    visited[state] = ng
                    parent[state] = (pos, t)
                    h = manhattan(pos, goal_pos)
                    node = AStarNode(pos=pos, time=next_t, g=ng, h=h)
                    heapq.heappush(open_heap, (node.f(), node))

        # Move
        for nb in grid.neighbors(pos):
            next_t = t + 1
            if next_t > time_horizon:
                continue
            if constraints.is_vertex_blocked(agent.agent_id, nb, next_t):
                continue
            if constraints.is_edge_blocked(agent.agent_id, pos, nb, next_t):
                continue

            state = (nb, next_t)
            ng = g + 1.0
            if ng < visited.get(state, math.inf):
                visited[state] = ng
                parent[state] = (pos, t)
                h = manhattan(nb, goal_pos)
                node = AStarNode(pos=nb, time=next_t, g=ng, h=h)
                heapq.heappush(open_heap, (node.f(), node))

    return None
