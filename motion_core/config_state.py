from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

from hardware.pca9685 import ServoLimits
from .math_utils import clamp_int


@dataclass(frozen=True)
class ConfigStateView:
    location_keys: list[str]
    channel_by_location: Dict[str, Optional[int]]
    servo_limits_by_location: Dict[str, ServoLimits]
    dynamic_limits: Dict[str, Any]

def servo_limits_from_config_item(item: Mapping[str, Any]) -> ServoLimits:
    lim = item.get("limits", {}) if isinstance(item.get("limits", {}), dict) else {}
    deg_min = clamp_int(int(lim.get("deg_min", 0)), 0, 270)
    deg_max = clamp_int(int(lim.get("deg_max", 270)), 0, 270)
    if deg_max < deg_min:
        deg_min, deg_max = deg_max, deg_min
    invert = bool(lim.get("invert", False))
    return ServoLimits(deg_min=deg_min, deg_max=deg_max, invert=invert)


def build_config_state_view(
    draft_cfg: Mapping[str, Any],
    dynamic_limits: Mapping[str, Any],
    location_keys: Iterable[str],
) -> ConfigStateView:
    locs = draft_cfg.get("locations", {}) if isinstance(draft_cfg.get("locations", {}), dict) else {}

    channel_by_location: Dict[str, Optional[int]] = {}
    limits_by_location: Dict[str, ServoLimits] = {}
    keys = [str(k) for k in location_keys]

    for loc_key in keys:
        item = locs.get(loc_key, {}) if isinstance(locs.get(loc_key, {}), dict) else {}

        ch: Optional[int] = None
        raw_ch = item.get("channel", None)
        if raw_ch is not None:
            try:
                ch_i = int(raw_ch)
                if 0 <= ch_i <= 15:
                    ch = ch_i
            except Exception:
                ch = None

        channel_by_location[loc_key] = ch
        limits_by_location[loc_key] = servo_limits_from_config_item(item)

    return ConfigStateView(
        location_keys=keys,
        channel_by_location=channel_by_location,
        servo_limits_by_location=limits_by_location,
        dynamic_limits=dict(dynamic_limits),
    )
