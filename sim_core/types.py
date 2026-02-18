from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 270

LOCATION_KEYS = [
    "front_left_hip",
    "front_left_knee",
    "front_right_hip",
    "front_right_knee",
    "rear_left_hip",
    "rear_left_knee",
    "rear_right_hip",
    "rear_right_knee",
]


@dataclass(frozen=True)
class Capsule:
    a: Tuple[float, float]
    b: Tuple[float, float]
    r: float
    name: str


@dataclass(frozen=True)
class CollisionDetails:
    pair: str
    min_distance_mm: float
    threshold_mm: float


@dataclass(frozen=True)
class CommandResult:
    requested_angle: int
    travel_applied_angle: int
    applied_angle: int
    clamped: bool
    clamp_reason: Optional[str]
    collision: Optional[Dict[str, Any]]
