from dataclasses import dataclass, field
from typing import List, Optional
from types_common import GridPos


@dataclass
class AgentTask:
    """Task satu agent dalam MG-MAPF.

    Attributes
    ----------
    agent_id : int
        ID unik agent (0, 1, 2, ...).
    start : GridPos
        Posisi awal agent.
    goals : List[GridPos]
        List posisi goal yang harus dikunjungi minimal sekali.
    """

    agent_id: int
    start: GridPos
    task_goals: List[GridPos]
    
    shelves: List[GridPos] = field(default_factory=list)       # rak yang jadi task
    loading_goal: Optional[GridPos] = None                     # loading station akhir
