from typing import Dict, List, Set
import math
from collections import deque
from types_common import GridPos
from grid import GridMap
from agent import AgentTask


def manhattan(a: GridPos, b: GridPos) -> int:
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by)


def compute_distance_table_for_agent(
    grid: GridMap,
    agent: AgentTask,
) -> Dict[GridPos, Dict[GridPos, int]]:
    """Tabel jarak statis (tanpa constraint) untuk heuristik MST."""
    relevant: Set[GridPos] = {agent.start, *agent.task_goals}
    dist_table: Dict[GridPos, Dict[GridPos, int]] = {}

    for src in relevant:
        q = deque()
        q.append(src)
        dist: Dict[GridPos, int] = {src: 0}

        while q:
            pos = q.popleft()
            for nb in grid.neighbors(pos):
                if nb not in dist:
                    dist[nb] = dist[pos] + 1
                    q.append(nb)

        dist_table[src] = dist

    return dist_table


def mst_heuristic(
    current_pos: GridPos,
    unvisited_goals: List[GridPos],
    dist_table: Dict[GridPos, Dict[GridPos, int]],
) -> float:
    """Estimasi biaya dengan MST heuristic (paper MGCBS)."""
    if not unvisited_goals:
        return 0.0

    nodes = [current_pos] + unvisited_goals
    n = len(nodes)

    in_mst = [False] * n
    key = [math.inf] * n
    key[0] = 0.0  # mulai dari current_pos
    total = 0.0

    for _ in range(n):
        u = -1
        min_val = math.inf
        for i in range(n):
            if not in_mst[i] and key[i] < min_val:
                min_val = key[i]
                u = i
        if u == -1:
            break
        in_mst[u] = True
        total += key[u]

        for v in range(n):
            if in_mst[v]:
                continue
            d = dist_table.get(nodes[u], {}).get(nodes[v], manhattan(nodes[u], nodes[v]))
            if d < key[v]:
                key[v] = d

    return total
