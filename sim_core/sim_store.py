from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .types import ANGLE_MAX_DEG, ANGLE_MIN_DEG, LOCATION_KEYS


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def coerce_float(v: Any, fallback: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(fallback)


def coerce_int(v: Any, fallback: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(fallback)


def default_dyn_side() -> Dict[str, Any]:
    return {
        "hip_dx_mm": 80.0,
        "hip_y_mm": 60.0,
        "thigh_len_front_mm": 70.0,
        "shin_len_front_mm": 90.0,
        "thigh_len_rear_mm": 70.0,
        "shin_len_rear_mm": 90.0,
        "front_knee_attach_backoff_mm": 0.0,
        "rear_knee_attach_backoff_mm": 0.0,
        "front_thigh_radius_mm": 6.0,
        "front_shin_radius_mm": 6.0,
        "rear_thigh_radius_mm": 6.0,
        "rear_shin_radius_mm": 6.0,
        "front_hip_offset_deg": 0.0,
        "front_knee_offset_deg": 0.0,
        "rear_hip_offset_deg": 0.0,
        "rear_knee_offset_deg": 0.0,
    }


def default_calibration_profile() -> Dict[str, Any]:
    return {
        "fit_mode": "linear_best_fit",
        "measurement_mode": "servo_physical_deg",
        "pairs": [
            {"commanded_deg": 0.0, "actual_deg": 0.0},
            {"commanded_deg": 270.0, "actual_deg": 270.0},
        ],
    }


def default_hip_standard_profile() -> Dict[str, Any]:
    return {
        "fit_mode": "linear_best_fit",
        "measurement_mode": "hip_line_relative_deg",
        "pairs": [
            {"commanded_deg": 163.0, "actual_deg": 90.0},
            {"commanded_deg": 135.0, "actual_deg": 135.0},
        ],
    }


def default_knee_standard_profile() -> Dict[str, Any]:
    return {
        "fit_mode": "linear_best_fit",
        "measurement_mode": "knee_relative_deg",
        "pairs": [
            {"commanded_deg": 163.0, "actual_deg": 90.0},
            {"commanded_deg": 30.0, "actual_deg": 180.0},
        ],
    }


def default_joint_calibration_map(location_keys: Optional[List[str]] = None) -> Dict[str, str]:
    keys = location_keys or LOCATION_KEYS
    out: Dict[str, str] = {}
    for loc_key in keys:
        out[loc_key] = "knee_standard" if str(loc_key).endswith("_knee") else "hip_standard"
    return out


def default_dyn_vars(location_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "version": 2,
        "active_side": "left",
        "search_step_deg": 1,
        "search_max_iters": 300,
        "left": default_dyn_side(),
        "right": default_dyn_side(),
        "calibration_profiles": {
            "identity": default_calibration_profile(),
            "hip_standard": default_hip_standard_profile(),
            "knee_standard": default_knee_standard_profile(),
        },
        "joint_calibration_map": default_joint_calibration_map(location_keys=location_keys),
    }


def validate_dyn_side_values(side_name: str, side_vals: Dict[str, Any]) -> Optional[str]:
    for key in ("thigh_len_front_mm", "shin_len_front_mm", "thigh_len_rear_mm", "shin_len_rear_mm"):
        if float(side_vals.get(key, 0.0)) <= 0.0:
            return f"{side_name}.{key} must be > 0"

    for key in ("front_thigh_radius_mm", "front_shin_radius_mm", "rear_thigh_radius_mm", "rear_shin_radius_mm"):
        if float(side_vals.get(key, 0.0)) <= 0.0:
            return f"{side_name}.{key} must be > 0"

    if float(side_vals.get("hip_dx_mm", 0.0)) < 0.0:
        return f"{side_name}.hip_dx_mm must be >= 0"

    front_backoff = float(side_vals.get("front_knee_attach_backoff_mm", 0.0))
    rear_backoff = float(side_vals.get("rear_knee_attach_backoff_mm", 0.0))
    front_thigh_len = float(side_vals.get("thigh_len_front_mm", 0.0))
    rear_thigh_len = float(side_vals.get("thigh_len_rear_mm", 0.0))

    if front_backoff < 0.0:
        return f"{side_name}.front_knee_attach_backoff_mm must be >= 0"
    if rear_backoff < 0.0:
        return f"{side_name}.rear_knee_attach_backoff_mm must be >= 0"
    if front_backoff > front_thigh_len:
        return f"{side_name}.front_knee_attach_backoff_mm must be <= {side_name}.thigh_len_front_mm"
    if rear_backoff > rear_thigh_len:
        return f"{side_name}.rear_knee_attach_backoff_mm must be <= {side_name}.thigh_len_rear_mm"

    return None


def load_dyn_vars(dyn_vars_file: Path, location_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    keys = location_keys or LOCATION_KEYS
    dyn_vars_file.parent.mkdir(parents=True, exist_ok=True)
    if not dyn_vars_file.exists():
        dv = default_dyn_vars(location_keys=keys)
        dyn_vars_file.write_text(json.dumps(dv, indent=2))
        return dv

    try:
        raw = json.loads(dyn_vars_file.read_text())
        out = default_dyn_vars(location_keys=keys)

        if int(raw.get("version", 0)) not in (1, 2):
            return out

        active_side = str(raw.get("active_side", "left")).strip().lower()
        if active_side not in ("left", "right"):
            active_side = "left"
        out["active_side"] = active_side

        out["search_step_deg"] = clamp_int(coerce_int(raw.get("search_step_deg", 1), 1), 1, 45)
        out["search_max_iters"] = clamp_int(coerce_int(raw.get("search_max_iters", 300), 300), 1, 5000)

        for side in ("left", "right"):
            side_raw = raw.get(side, {})
            if not isinstance(side_raw, dict):
                continue
            side_out = out[side]
            for k in list(side_out.keys()):
                if k.endswith("_mm") or k.endswith("_deg") or "radius" in k or "len" in k or "hip_" in k:
                    side_out[k] = coerce_float(side_raw.get(k, side_out[k]), side_out[k])

        profiles_out: Dict[str, Dict[str, Any]] = {}
        profiles_raw = raw.get("calibration_profiles", {})
        if isinstance(profiles_raw, dict):
            for name_raw, prof_raw in profiles_raw.items():
                name = str(name_raw).strip()
                if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name) is None:
                    continue
                if not isinstance(prof_raw, dict):
                    continue
                mode = str(prof_raw.get("measurement_mode", "servo_physical_deg")).strip().lower()
                if mode not in ("servo_physical_deg", "hip_line_relative_deg", "knee_relative_deg"):
                    mode = "servo_physical_deg"
                pairs_in = prof_raw.get("pairs", [])
                pairs_out: List[Dict[str, float]] = []
                if isinstance(pairs_in, list):
                    for item in pairs_in:
                        if not isinstance(item, dict):
                            continue
                        c = coerce_float(item.get("commanded_deg", 0.0), 0.0)
                        a = coerce_float(item.get("actual_deg", 0.0), 0.0)
                        pairs_out.append(
                            {
                                "commanded_deg": float(max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, c))),
                                "actual_deg": float(max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, a))),
                            }
                        )
                profiles_out[name] = {
                    "fit_mode": "linear_best_fit",
                    "measurement_mode": mode,
                    "pairs": pairs_out,
                }

        if "identity" not in profiles_out:
            profiles_out["identity"] = default_calibration_profile()
        if "hip_standard" not in profiles_out:
            profiles_out["hip_standard"] = default_hip_standard_profile()
        if "knee_standard" not in profiles_out:
            profiles_out["knee_standard"] = default_knee_standard_profile()
        out["calibration_profiles"] = profiles_out

        map_out = default_joint_calibration_map(location_keys=keys)
        map_raw = raw.get("joint_calibration_map", {})
        if isinstance(map_raw, dict):
            for loc_key in keys:
                assigned = str(map_raw.get(loc_key, map_out[loc_key])).strip()
                if assigned in profiles_out:
                    map_out[loc_key] = assigned
                else:
                    map_out[loc_key] = "identity"
        out["joint_calibration_map"] = map_out

        return out
    except Exception:
        return default_dyn_vars(location_keys=keys)


def save_dyn_vars(dyn_vars_file: Path, dv: Dict[str, Any]) -> None:
    dyn_vars_file.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(dv)
    payload["version"] = 2
    dyn_vars_file.write_text(json.dumps(payload, indent=2))


def stance_location_keys(location_keys: Optional[List[str]] = None) -> List[str]:
    return list(location_keys or LOCATION_KEYS)


def validate_stance_name(raw: Any) -> Tuple[bool, str]:
    name = str(raw or "").strip()
    if not name:
        return False, ""
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        return False, ""
    return True, name


def load_stance(stance_dir: Path, name: str, location_keys: Optional[List[str]] = None) -> Tuple[bool, Dict[str, int], str]:
    keys = stance_location_keys(location_keys=location_keys)
    ok_name, clean_name = validate_stance_name(name)
    if not ok_name:
        return False, {}, "stance name must match [A-Za-z0-9_-]+"

    stance_dir.mkdir(parents=True, exist_ok=True)
    path = stance_dir / f"{clean_name}.json"
    if not path.exists():
        return False, {}, f"stance not found: {clean_name}"

    try:
        raw = json.loads(path.read_text())
    except Exception as e:
        return False, {}, f"failed to read stance '{clean_name}': {e}"

    if not isinstance(raw, dict):
        return False, {}, "stance file must be an object"
    angles = raw.get("angles", None)
    if not isinstance(angles, dict):
        return False, {}, "stance file must include 'angles' object"

    out: Dict[str, int] = {}
    missing: List[str] = []
    for key in keys:
        if key not in angles:
            missing.append(key)
            continue
        try:
            val = int(angles[key])
        except Exception:
            return False, {}, f"stance '{clean_name}' angle '{key}' must be an integer"
        if not (ANGLE_MIN_DEG <= val <= ANGLE_MAX_DEG):
            return False, {}, f"stance '{clean_name}' angle '{key}' out of range {ANGLE_MIN_DEG}..{ANGLE_MAX_DEG}"
        out[key] = val

    if missing:
        return False, {}, f"stance '{clean_name}' missing keys: {missing}"

    return True, out, ""
