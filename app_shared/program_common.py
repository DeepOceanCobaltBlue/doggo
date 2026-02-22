from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class Location:
    key: str
    label: str


DOGGO_LOCATIONS: List[Location] = [
    Location("front_left_hip", "Front Left Hip"),
    Location("front_left_knee", "Front Left Knee"),
    Location("front_right_hip", "Front Right Hip"),
    Location("front_right_knee", "Front Right Knee"),
    Location("rear_left_hip", "Rear Left Hip"),
    Location("rear_left_knee", "Rear Left Knee"),
    Location("rear_right_hip", "Rear Right Hip"),
    Location("rear_right_knee", "Rear Right Knee"),
]


def read_json_file(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def validate_location_order(order: Any) -> Tuple[bool, List[str], str]:
    if not isinstance(order, list):
        return False, [], "location_order.position_order must be a list"
    cleaned = [str(x).strip() for x in order if str(x).strip()]
    if len(cleaned) != 8:
        return False, [], "position_order must contain exactly 8 location keys"
    if len(set(cleaned)) != 8:
        return False, [], "position_order contains duplicate keys"
    return True, cleaned, ""


def summarize_timeline(timeline: Any) -> Dict[str, Any]:
    return {
        "program_id": timeline.program_id,
        "tick_ms": timeline.tick_ms,
        "duration_ms": timeline.duration_ms,
        "num_ticks": len(timeline.ticks),
        "location_keys": list(timeline.location_keys),
    }

