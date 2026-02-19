from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


VALID_EASING = {"linear", "ease_in", "ease_out", "ease_in_out"}


@dataclass(frozen=True)
class ProgramCommand:
    location: str
    target_angle: int
    duration_ms: int
    easing: str = "linear"
    max_deg_per_sec: Optional[float] = None


@dataclass(frozen=True)
class ProgramStep:
    step_id: str
    commands: List[ProgramCommand] = field(default_factory=list)
    advance: str = "all_complete"


@dataclass(frozen=True)
class ProgramSpec:
    program_id: str
    tick_ms: int
    steps: List[ProgramStep] = field(default_factory=list)


@dataclass(frozen=True)
class TickMeta:
    step_id: str
    step_index: int


@dataclass(frozen=True)
class TimelineTick:
    tick: int
    t_ms: int
    targets: Dict[str, int]
    meta: TickMeta


@dataclass(frozen=True)
class CommandPacket:
    tick: int
    t_ms: int
    targets: Dict[str, int]


@dataclass(frozen=True)
class CompiledTimeline:
    program_id: str
    tick_ms: int
    duration_ms: int
    location_keys: List[str]
    ticks: List[TimelineTick]
