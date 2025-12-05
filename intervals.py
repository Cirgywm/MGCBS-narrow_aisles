from dataclasses import dataclass
from typing import Dict, List
from types_common import GridPos, Time
from grid import GridMap
from agent import AgentTask
from constraints import ConstraintTable


@dataclass
class SafeInterval:
    """Safe Interval (SI) di satu vertex: [start_t, end_t] inklusif."""
    vertex: GridPos
    start_t: Time
    end_t: Time

    def contains(self, t: Time) -> bool:
        return self.start_t <= t <= self.end_t


@dataclass
class GoalSafeInterval:
    """Goal Safe Interval (GSI) di sebuah goal vertex."""
    goal_index: int
    interval: SafeInterval

    @property
    def vertex(self) -> GridPos:
        return self.interval.vertex

    @property
    def start_t(self) -> Time:
        return self.interval.start_t

    @property
    def end_t(self) -> Time:
        return self.interval.end_t


def build_safe_intervals_for_agent(
    grid: GridMap,
    agent: AgentTask,
    constraints: ConstraintTable,
    time_horizon: Time,
) -> Dict[GridPos, List[SafeInterval]]:
    """Bangun semua Safe Interval (SI) untuk setiap vertex pada agent ini."""
    si_by_vertex: Dict[GridPos, List[SafeInterval]] = {}

    for x in range(grid.width):
        for y in range(grid.height):
            v = (x, y)
            if not grid.passable(v):
                continue

            intervals: List[SafeInterval] = []
            in_interval = False
            start_t = 0

            for t in range(0, time_horizon + 1):
                blocked = constraints.is_vertex_blocked(agent.agent_id, v, t)
                if not blocked and not in_interval:
                    in_interval = True
                    start_t = t
                elif blocked and in_interval:
                    intervals.append(SafeInterval(v, start_t, t - 1))
                    in_interval = False

            if in_interval:
                intervals.append(SafeInterval(v, start_t, time_horizon))

            if intervals:
                si_by_vertex[v] = intervals

    return si_by_vertex


def build_goal_safe_intervals(
    agent: AgentTask,
    si_by_vertex: Dict[GridPos, List[SafeInterval]],
) -> Dict[int, List[GoalSafeInterval]]:
    """Bangun semua Goal Safe Interval (GSI) untuk setiap goal agent."""
    gsi_by_goal: Dict[int, List[GoalSafeInterval]] = {}
    for goal_idx, gpos in enumerate(agent.task_goals):
        si_list = si_by_vertex.get(gpos, [])
        gsi_by_goal[goal_idx] = [GoalSafeInterval(goal_idx, si) for si in si_list]
    return gsi_by_goal
