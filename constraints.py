from dataclasses import dataclass, field
from typing import Dict, Tuple
from types_common import GridPos, Time


@dataclass(frozen=True)
class VertexConstraint:
    """Constraint: agent tidak boleh berada di vertex pada waktu tertentu."""
    agent_id: int
    pos: GridPos
    time: Time


@dataclass(frozen=True)
class EdgeConstraint:
    """Constraint: agent tidak boleh melewati edge (u->v) pada waktu tertentu."""
    agent_id: int
    from_pos: GridPos
    to_pos: GridPos
    time: Time  # edge dipakai dari time-1 ke time


@dataclass
class ConstraintTable:
    """Menyimpan semua constraint (vertex dan edge) untuk seluruh agent.

    Implementasi sederhana dengan dictionary per agent agar lookup cepat.
    """

    vertex_constraints: Dict[int, Dict[Tuple[GridPos, Time], bool]] = field(
        default_factory=dict
    )
    edge_constraints: Dict[int, Dict[Tuple[GridPos, GridPos, Time], bool]] = field(
        default_factory=dict
    )

    def add_vertex_constraint(self, c: VertexConstraint) -> None:
        vc = self.vertex_constraints.setdefault(c.agent_id, {})
        vc[(c.pos, c.time)] = True

    def add_edge_constraint(self, c: EdgeConstraint) -> None:
        ec = self.edge_constraints.setdefault(c.agent_id, {})
        ec[(c.from_pos, c.to_pos, c.time)] = True

    def is_vertex_blocked(self, agent_id: int, pos: GridPos, t: Time) -> bool:
        return (pos, t) in self.vertex_constraints.get(agent_id, {})

    def is_edge_blocked(
        self,
        agent_id: int,
        from_pos: GridPos,
        to_pos: GridPos,
        t: Time,
    ) -> bool:
        return (from_pos, to_pos, t) in self.edge_constraints.get(agent_id, {})
