import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dataclasses import dataclass


@dataclass
class SearchMetrics:
    # High-level CBS
    cbs_nodes_expanded: int = 0

    # Low-level DSS (MGCBS)
    low_level_states_expanded: int = 0

    # Low-level A* ke GSI (A2)
    low_level_astar_expanded: int = 0

    # Konflik
    conflicts_found: int = 0
    conflicts_resolved: int = 0

    def total_nodes_expanded(self) -> int:
        """Total node yang di-expand di semua level search."""
        return (
            self.cbs_nodes_expanded
            + self.low_level_states_expanded
            + self.low_level_astar_expanded
        )
