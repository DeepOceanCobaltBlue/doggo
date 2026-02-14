from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

from flask import Flask, jsonify, request, send_from_directory

from hardware.pca9685 import PCA9685, ServoLimits


# -----------------------------
# Doggo constants (config app)
# -----------------------------
ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 270

CHANNEL_MIN = 0
CHANNEL_MAX = 15

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "config_file.json"
LOCATION_ORDER_FILE = CONFIG_DIR / "location_order.json"
CHANNEL_NOTES_FILE = CONFIG_DIR / "channel_notes.json"
CHANNEL_NOTES_MAX_CHARS = 50_000

DYN_VARS_FILE = CONFIG_DIR / "dynamic_limits.json"
STANCES_DIR = BASE_DIR / "stances"
DEFAULT_STANCE_NAME = "default"
STANCE_STEP_DELAY_MS = 250

# PCA9685 constants (locked frequency is inside driver)
I2C_BUS = 1
I2C_ADDR = 0x40


# -----------------------------
# Fixed locations (backend-owned)
# -----------------------------
@dataclass(frozen=True)
class Location:
    key: str
    label: str


LOCATIONS: List[Location] = [
    Location("front_left_hip", "Front Left Hip"),
    Location("front_left_knee", "Front Left Knee"),
    Location("front_right_hip", "Front Right Hip"),
    Location("front_right_knee", "Front Right Knee"),
    Location("rear_left_hip", "Rear Left Hip"),
    Location("rear_left_knee", "Rear Left Knee"),
    Location("rear_right_hip", "Rear Right Hip"),
    Location("rear_right_knee", "Rear Right Knee"),
]


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _all_channels() -> List[int]:
    return list(range(CHANNEL_MIN, CHANNEL_MAX + 1))


def _default_limits() -> Dict[str, Any]:
    return {"deg_min": 0, "deg_max": 270, "invert": False}


def _empty_config() -> Dict[str, Any]:
    """
    Canonical config structure (version 2):
    {
      "version": 2,
      "locations": {
        "<loc_key>": {
          "channel": <int or null>,
          "limits": { "deg_min": int, "deg_max": int, "invert": bool }
        },
        ...
      }
    }
    """
    return {
        "version": 2,
        "locations": {
            loc.key: {"channel": None, "limits": _default_limits()}
            for loc in LOCATIONS
        },
    }


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_init_location_order() -> List[str]:
    _ensure_config_dir()

    default_order = [loc.key for loc in LOCATIONS]

    if not LOCATION_ORDER_FILE.exists():
        LOCATION_ORDER_FILE.write_text(json.dumps({"version": 1, "position_order": default_order}, indent=2))
        return default_order

    try:
        data = json.loads(LOCATION_ORDER_FILE.read_text())
        order = data.get("position_order", [])
        if not isinstance(order, list):
            return default_order
        order = [str(x) for x in order]

        known = {loc.key for loc in LOCATIONS}
        cleaned = [k for k in order if k in known]
        for k in default_order:
            if k not in cleaned:
                cleaned.append(k)

        if cleaned != order:
            LOCATION_ORDER_FILE.write_text(json.dumps({"version": 1, "position_order": cleaned}, indent=2))
        return cleaned
    except Exception:
        LOCATION_ORDER_FILE.write_text(json.dumps({"version": 1, "position_order": default_order}, indent=2))
        return default_order


def _load_saved_config() -> Dict[str, Any]:
    _ensure_config_dir()

    if not CONFIG_FILE.exists():
        return _empty_config()

    try:
        data = json.loads(CONFIG_FILE.read_text())
        if int(data.get("version", 0)) != 2:
            return _empty_config()

        out = _empty_config()
        locs = data.get("locations", {})
        if not isinstance(locs, dict):
            return out

        for loc in LOCATIONS:
            item = locs.get(loc.key, {})
            if not isinstance(item, dict):
                continue

            # channel
            ch = item.get("channel", None)
            if ch is None:
                out["locations"][loc.key]["channel"] = None
            else:
                try:
                    ch_i = int(ch)
                    if CHANNEL_MIN <= ch_i <= CHANNEL_MAX:
                        out["locations"][loc.key]["channel"] = ch_i
                    else:
                        out["locations"][loc.key]["channel"] = None
                except Exception:
                    out["locations"][loc.key]["channel"] = None

            # limits
            lim = item.get("limits", {})
            if not isinstance(lim, dict):
                lim = {}

            try:
                raw_min = int(lim.get("deg_min", 0))
            except Exception:
                raw_min = 0
            try:
                raw_max = int(lim.get("deg_max", 270))
            except Exception:
                raw_max = 270

            deg_min = _clamp_int(raw_min, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
            deg_max = _clamp_int(raw_max, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
            if deg_max < deg_min:
                deg_min, deg_max = deg_max, deg_min
            invert = bool(lim.get("invert", False))

            out["locations"][loc.key]["limits"] = {"deg_min": deg_min, "deg_max": deg_max, "invert": invert}

        return out
    except Exception:
        return _empty_config()


def _save_config(cfg: Dict[str, Any]) -> None:
    _ensure_config_dir()
    cfg = dict(cfg)
    cfg["version"] = 2
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def _compute_available_channels(cfg: Dict[str, Any]) -> List[int]:
    used = set()
    for loc in LOCATIONS:
        ch = cfg["locations"][loc.key]["channel"]
        if ch is not None:
            used.add(int(ch))
    return [ch for ch in _all_channels() if ch not in used]


def _dirty(draft: Dict[str, Any], saved: Dict[str, Any]) -> bool:
    return draft != saved


def _validate_no_duplicate_channels(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    seen: Dict[int, str] = {}
    for loc in LOCATIONS:
        ch = cfg["locations"][loc.key]["channel"]
        if ch is None:
            continue
        ch_i = int(ch)
        if ch_i in seen:
            return False, f"Duplicate channel {ch_i} assigned to '{seen[ch_i]}' and '{loc.key}'."
        seen[ch_i] = loc.key
    return True, ""


def _get_limits(cfg: Dict[str, Any], loc_key: str) -> ServoLimits:
    lim = cfg["locations"][loc_key]["limits"]
    return ServoLimits(
        deg_min=int(lim["deg_min"]),
        deg_max=int(lim["deg_max"]),
        invert=bool(lim["invert"]),
    )


# -----------------------------
# Channel notes helpers
# -----------------------------
def _load_channel_notes() -> str:
    _ensure_config_dir()
    if not CHANNEL_NOTES_FILE.exists():
        return ""
    try:
        data = json.loads(CHANNEL_NOTES_FILE.read_text())
        notes = data.get("notes", "")
        if not isinstance(notes, str):
            return ""
        return notes
    except Exception:
        return ""


def _save_channel_notes(notes: str) -> None:
    _ensure_config_dir()
    payload = {"version": 1, "notes": notes}
    CHANNEL_NOTES_FILE.write_text(json.dumps(payload, indent=2))


# -----------------------------
# Dynamic limit variables (capsule model)
# -----------------------------
def _default_dyn_side() -> Dict[str, Any]:
    # These defaults are intentionally "generic" until you calibrate.
    return {
        "hip_dx_mm": 80.0,
        "hip_y_mm": 60.0,
        "thigh_len_front_mm": 70.0,
        "shin_len_front_mm": 90.0,
        "thigh_len_rear_mm": 70.0,
        "shin_len_rear_mm": 90.0,
        "front_thigh_radius_mm": 6.0,
        "front_shin_radius_mm": 6.0,
        "rear_thigh_radius_mm": 6.0,
        "rear_shin_radius_mm": 6.0,
        "front_hip_offset_deg": 0.0,
        "front_knee_offset_deg": 0.0,
        "rear_hip_offset_deg": 0.0,
        "rear_knee_offset_deg": 0.0,
    }


def _default_dyn_vars() -> Dict[str, Any]:
    return {
        "version": 1,
        "active_side": "left",
        "search_step_deg": 1,
        "search_max_iters": 300,
        "left": _default_dyn_side(),
        "right": _default_dyn_side(),
    }


def _coerce_float(v: Any, fallback: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(fallback)


def _coerce_int(v: Any, fallback: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(fallback)


def _load_dyn_vars() -> Dict[str, Any]:
    _ensure_config_dir()
    if not DYN_VARS_FILE.exists():
        dv = _default_dyn_vars()
        DYN_VARS_FILE.write_text(json.dumps(dv, indent=2))
        return dv

    try:
        raw = json.loads(DYN_VARS_FILE.read_text())
        out = _default_dyn_vars()

        if int(raw.get("version", 0)) != 1:
            return out

        active_side = str(raw.get("active_side", "left")).strip().lower()
        if active_side not in ("left", "right"):
            active_side = "left"
        out["active_side"] = active_side

        out["search_step_deg"] = _clamp_int(_coerce_int(raw.get("search_step_deg", 1), 1), 1, 45)
        out["search_max_iters"] = _clamp_int(_coerce_int(raw.get("search_max_iters", 300), 300), 1, 5000)

        for side in ("left", "right"):
            side_raw = raw.get(side, {})
            if not isinstance(side_raw, dict):
                continue
            side_out = out[side]
            for k in list(side_out.keys()):
                if k.endswith("_mm") or k.endswith("_deg") or "radius" in k or "len" in k or "hip_" in k:
                    side_out[k] = _coerce_float(side_raw.get(k, side_out[k]), side_out[k])

        return out
    except Exception:
        return _default_dyn_vars()


def _save_dyn_vars(dv: Dict[str, Any]) -> None:
    _ensure_config_dir()
    dv = dict(dv)
    dv["version"] = 1
    DYN_VARS_FILE.write_text(json.dumps(dv, indent=2))


def _validate_dyn_side_values(side_name: str, side_vals: Dict[str, Any]) -> Optional[str]:
    for key in ("thigh_len_front_mm", "shin_len_front_mm", "thigh_len_rear_mm", "shin_len_rear_mm"):
        if float(side_vals.get(key, 0.0)) <= 0.0:
            return f"{side_name}.{key} must be > 0"

    for key in ("front_thigh_radius_mm", "front_shin_radius_mm", "rear_thigh_radius_mm", "rear_shin_radius_mm"):
        if float(side_vals.get(key, 0.0)) <= 0.0:
            return f"{side_name}.{key} must be > 0"

    if float(side_vals.get("hip_dx_mm", 0.0)) < 0.0:
        return f"{side_name}.hip_dx_mm must be >= 0"

    return None


# -----------------------------
# Stance helpers
# -----------------------------
def _ensure_stances_dir() -> None:
    STANCES_DIR.mkdir(parents=True, exist_ok=True)


def _stance_location_keys() -> List[str]:
    return [loc.key for loc in LOCATIONS]


def _validate_stance_name(raw: Any) -> Tuple[bool, str]:
    name = str(raw or "").strip()
    if not name:
        return False, ""
    if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
        return False, ""
    return True, name


def _load_stance(name: str) -> Tuple[bool, Dict[str, int], str]:
    ok_name, clean_name = _validate_stance_name(name)
    if not ok_name:
        return False, {}, "stance name must match [A-Za-z0-9_-]+"

    _ensure_stances_dir()
    path = STANCES_DIR / f"{clean_name}.json"
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
    for key in _stance_location_keys():
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


# -----------------------------
# Collision math (2D capsules)
# -----------------------------
def _deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def _unit_from_angle_deg(deg: float) -> Tuple[float, float]:
    r = _deg_to_rad(deg)
    # Screen/robot convention: positive angle rotates clockwise in x-right, y-up model.
    return (math.cos(r), -math.sin(r))


def _pt_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def _pt_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _pt_mul(a: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (a[0] * s, a[1] * s)


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _norm2(a: Tuple[float, float]) -> float:
    return _dot(a, a)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _dist_point_to_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ab = _pt_sub(b, a)
    ap = _pt_sub(p, a)
    denom = _norm2(ab)
    if denom <= 1e-12:
        return _dist(p, a)
    t = _dot(ap, ab) / denom
    t = max(0.0, min(1.0, t))
    proj = _pt_add(a, _pt_mul(ab, t))
    return _dist(p, proj)


def _segments_intersect(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    # Robust-ish 2D segment intersection via orientation tests
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r) -> bool:
        return (min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
                min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9)

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # General case
    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    # Colinear cases
    if abs(o1) <= 1e-9 and on_seg(a, c, b):
        return True
    if abs(o2) <= 1e-9 and on_seg(a, d, b):
        return True
    if abs(o3) <= 1e-9 and on_seg(c, a, d):
        return True
    if abs(o4) <= 1e-9 and on_seg(c, b, d):
        return True

    return False


def _dist_segment_to_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _dist_point_to_segment(a, c, d),
        _dist_point_to_segment(b, c, d),
        _dist_point_to_segment(c, a, b),
        _dist_point_to_segment(d, a, b),
    )


@dataclass(frozen=True)
class Capsule:
    a: Tuple[float, float]
    b: Tuple[float, float]
    r: float
    name: str  # for diagnostics


def _capsules_min_distance(c1: Capsule, c2: Capsule) -> float:
    return _dist_segment_to_segment(c1.a, c1.b, c2.a, c2.b)


def _infer_side(loc_key: str) -> str:
    if "_left_" in loc_key or loc_key.endswith("_left") or "left" in loc_key:
        return "left"
    if "_right_" in loc_key or loc_key.endswith("_right") or "right" in loc_key:
        return "right"
    # Fallback: treat unknown as left
    return "left"


def _normalize_deg(deg: float) -> float:
    # Keep in 0..360 for trig; no need to wrap to 0..270 here
    x = deg % 360.0
    if x < 0:
        x += 360.0
    return x


def _build_leg_capsules_for_side(
    dv: Dict[str, Any],
    side: str,
    angles: Dict[str, float],
) -> Tuple[List[Capsule], List[Capsule]]:
    """
    Build capsules for FRONT and REAR leg for the given side.

    Coordinate system:
      x: forward (front positive)
      y: up

    Anchors:
      front hip at (0, hip_y)
      rear hip at:
        - (+hip_dx, hip_y) for left side (front appears left of rear in view)
        - (-hip_dx, hip_y) for right side (front appears right of rear in view)

    Angles are absolute directions (servo-angle terms + offsets):
      - hip angle -> thigh direction
      - knee angle -> shin direction
    """
    s = dv[side]
    hip_dx = float(s["hip_dx_mm"])
    hip_y = float(s["hip_y_mm"])

    # Anchors
    Hf = (0.0, hip_y)
    Hr = (hip_dx, hip_y) if side == "left" else (-hip_dx, hip_y)

    # Lengths
    Ltf = float(s["thigh_len_front_mm"])
    Lsf = float(s["shin_len_front_mm"])
    Ltr = float(s["thigh_len_rear_mm"])
    Lsr = float(s["shin_len_rear_mm"])

    # Radii
    r_f_thigh = float(s["front_thigh_radius_mm"])
    r_f_shin = float(s["front_shin_radius_mm"])
    r_r_thigh = float(s["rear_thigh_radius_mm"])
    r_r_shin = float(s["rear_shin_radius_mm"])

    # Offsets
    off_f_hip = float(s["front_hip_offset_deg"])
    off_f_knee = float(s["front_knee_offset_deg"])
    off_r_hip = float(s["rear_hip_offset_deg"])
    off_r_knee = float(s["rear_knee_offset_deg"])

    # Read commanded/known angles (0..270), convert to sim absolute directions
    a_f_hip_cmd = float(angles.get("front_hip", 135.0))
    a_f_knee_cmd = float(angles.get("front_knee", 135.0))
    a_r_hip_cmd = float(angles.get("rear_hip", 135.0))
    a_r_knee_cmd = float(angles.get("rear_knee", 135.0))

    a_f_hip = _normalize_deg(a_f_hip_cmd + off_f_hip)
    a_f_knee = _normalize_deg(a_f_knee_cmd + off_f_knee)
    a_r_hip = _normalize_deg(a_r_hip_cmd + off_r_hip)
    a_r_knee = _normalize_deg(a_r_knee_cmd + off_r_knee)

    # FRONT: hip -> knee
    uf = _unit_from_angle_deg(a_f_hip)
    Kf = _pt_add(Hf, _pt_mul(uf, Ltf))
    # FRONT: knee -> foot
    vf = _unit_from_angle_deg(a_f_knee)
    Ff = _pt_add(Kf, _pt_mul(vf, Lsf))

    # REAR: hip -> knee
    ur = _unit_from_angle_deg(a_r_hip)
    Kr = _pt_add(Hr, _pt_mul(ur, Ltr))
    # REAR: knee -> foot
    vr = _unit_from_angle_deg(a_r_knee)
    Fr = _pt_add(Kr, _pt_mul(vr, Lsr))

    front_caps = [
        Capsule(a=Hf, b=Kf, r=r_f_thigh, name="front_thigh"),
        Capsule(a=Kf, b=Ff, r=r_f_shin, name="front_shin"),
    ]
    rear_caps = [
        Capsule(a=Hr, b=Kr, r=r_r_thigh, name="rear_thigh"),
        Capsule(a=Kr, b=Fr, r=r_r_shin, name="rear_shin"),
    ]
    return front_caps, rear_caps


def _angles_pack_for_side_from_state(side: str, state_angles: Dict[str, int]) -> Dict[str, float]:
    """
    Extract 4 joint angles for the side from the full location-key->angle dict.
    """
    if side == "left":
        return {
            "front_hip": float(state_angles.get("front_left_hip", 135)),
            "front_knee": float(state_angles.get("front_left_knee", 135)),
            "rear_hip": float(state_angles.get("rear_left_hip", 135)),
            "rear_knee": float(state_angles.get("rear_left_knee", 135)),
        }
    else:
        return {
            "front_hip": float(state_angles.get("front_right_hip", 135)),
            "front_knee": float(state_angles.get("front_right_knee", 135)),
            "rear_hip": float(state_angles.get("rear_right_hip", 135)),
            "rear_knee": float(state_angles.get("rear_right_knee", 135)),
        }


def _predict_collision_for_side(
    dv: Dict[str, Any],
    side: str,
    state_angles: Dict[str, int],
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Returns (collides, details) for the current pose on a given side.
    Details include closest pair info.
    """
    pack = _angles_pack_for_side_from_state(side, state_angles)
    front_caps, rear_caps = _build_leg_capsules_for_side(dv, side, pack)

    best = None  # (min_dist, threshold, c1.name, c2.name)
    for c1 in front_caps:
        for c2 in rear_caps:
            dmin = _capsules_min_distance(c1, c2)
            thresh = float(c1.r + c2.r)
            if best is None or dmin < best[0]:
                best = (dmin, thresh, c1.name, c2.name)

    if best is None:
        return False, None

    dmin, thresh, n1, n2 = best
    if dmin <= thresh:
        return True, {
            "pair": f"{n1} vs {n2}",
            "min_distance_mm": dmin,
            "threshold_mm": thresh,
        }
    return False, {
        "pair": f"{n1} vs {n2}",
        "min_distance_mm": dmin,
        "threshold_mm": thresh,
    }


def _apply_angle_to_state(loc_key: str, angle: int, state_angles: Dict[str, int]) -> Dict[str, int]:
    nxt = dict(state_angles)
    nxt[loc_key] = angle
    return nxt


def _solve_closest_safe_angle_step_search(
    dv: Dict[str, Any],
    side: str,
    loc_key: str,
    requested_angle: int,
    current_angle: int,
    state_angles: Dict[str, int],
    step_deg: int,
    max_iters: int,
    angle_lo: int = ANGLE_MIN_DEG,
    angle_hi: int = ANGLE_MAX_DEG,
) -> Tuple[int, bool, Optional[Dict[str, Any]]]:
    """
    If requested is safe -> return (requested, False, details).
    Else search nearest safe candidates around requested in step increments.
    Preference order for equal-distance candidates is toward current angle first.
    Returns (applied, clamped, details_for_applied_or_last_checked).
    """
    requested_angle = int(requested_angle)
    current_angle = int(current_angle)
    step_deg = max(1, int(step_deg))
    max_iters = max(1, int(max_iters))
    lo = int(min(angle_lo, angle_hi))
    hi = int(max(angle_lo, angle_hi))
    requested_angle = _clamp_int(requested_angle, lo, hi)
    current_angle = _clamp_int(current_angle, lo, hi)

    # Quick check requested
    test_state = _apply_angle_to_state(loc_key, requested_angle, state_angles)
    collides, details = _predict_collision_for_side(dv, side, test_state)
    if not collides:
        return requested_angle, False, details

    best_details = details
    dir_to_current = -1 if current_angle < requested_angle else +1 if current_angle > requested_angle else 0

    for i in range(1, max_iters + 1):
        delta = i * step_deg
        candidates: List[int] = []
        if dir_to_current == 0:
            candidates = [requested_angle - delta, requested_angle + delta]
        else:
            # Prefer equal-distance candidate toward current first.
            candidates = [requested_angle + dir_to_current * delta, requested_angle - dir_to_current * delta]

        in_range = [c for c in candidates if lo <= c <= hi]
        if not in_range:
            if (requested_angle - delta) < lo and (requested_angle + delta) > hi:
                break
            continue

        for candidate in in_range:
            test_state = _apply_angle_to_state(loc_key, int(candidate), state_angles)
            collides, det = _predict_collision_for_side(dv, side, test_state)
            best_details = det
            if not collides:
                return int(candidate), True, det

    # Could not find safe near requested; stick with current (even if colliding)
    fallback = _clamp_int(int(current_angle), lo, hi)
    test_state = _apply_angle_to_state(loc_key, fallback, state_angles)
    _, det = _predict_collision_for_side(dv, side, test_state)
    return fallback, True, det


# -----------------------------
# App init
# -----------------------------
app = Flask(__name__, static_folder="static")

_location_order: List[str] = _load_or_init_location_order()

_saved_cfg: Dict[str, Any] = _load_saved_config()
_draft_cfg: Dict[str, Any] = json.loads(json.dumps(_saved_cfg))

_dyn_vars: Dict[str, Any] = _load_dyn_vars()

# Hardware driver (optional)
pca: Optional[PCA9685] = None
_pca_error: Optional[str] = None
try:
    pca = PCA9685(i2c_bus=I2C_BUS, address=I2C_ADDR)
except Exception as e:
    pca = None
    _pca_error = str(e)

# Angle state
DEFAULT_NEUTRAL = 135
_hw_angles: Dict[str, int] = {loc.key: DEFAULT_NEUTRAL for loc in LOCATIONS}
_sim_angles: Dict[str, int] = {loc.key: DEFAULT_NEUTRAL for loc in LOCATIONS}


# -----------------------------
# Static UI
# -----------------------------
@app.get("/")
def index():
    return send_from_directory("static", "config_page.html")


def _execute_servo_command(loc_key: str, requested_angle: int, mode: str) -> Tuple[bool, Dict[str, Any], int]:
    global _draft_cfg, pca, _dyn_vars, _hw_angles, _sim_angles

    if loc_key not in _draft_cfg["locations"]:
        return False, {"ok": False, "error": f"Unknown location '{loc_key}'"}, 400

    if mode not in ("normal", "test"):
        mode = "normal"

    requested = _clamp_int(int(requested_angle), ANGLE_MIN_DEG, ANGLE_MAX_DEG)

    # Apply travel clamp/invert (in degree space) using ServoLimits
    limits = _get_limits(_draft_cfg, loc_key)
    deg_min = _clamp_int(int(limits.deg_min), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    deg_max = _clamp_int(int(limits.deg_max), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    if deg_max < deg_min:
        deg_min, deg_max = deg_max, deg_min

    travel_applied = _clamp_int(int(requested), deg_min, deg_max)
    if limits.invert:
        travel_applied = deg_max - (travel_applied - deg_min)
        travel_applied = _clamp_int(int(travel_applied), deg_min, deg_max)

    # Decide which state we're walking against for step search
    state_angles = _sim_angles if mode == "test" else _hw_angles
    current_angle = int(state_angles.get(loc_key, DEFAULT_NEUTRAL))

    # Collision clamp
    side = _infer_side(loc_key)
    step_deg = _clamp_int(_coerce_int(_dyn_vars.get("search_step_deg", 1), 1), 1, 45)
    max_iters = _clamp_int(_coerce_int(_dyn_vars.get("search_max_iters", 300), 300), 1, 5000)

    applied, clamped, collision_details = _solve_closest_safe_angle_step_search(
        dv=_dyn_vars,
        side=side,
        loc_key=loc_key,
        requested_angle=int(travel_applied),
        current_angle=int(current_angle),
        state_angles=state_angles,
        step_deg=step_deg,
        max_iters=max_iters,
        angle_lo=deg_min,
        angle_hi=deg_max,
    )

    # In normal mode, require channel + hardware
    if mode == "normal":
        ch = _draft_cfg["locations"][loc_key]["channel"]
        if ch is None:
            return False, {"ok": False, "error": f"Location '{loc_key}' is unassigned"}, 409
        if pca is None:
            return False, {"ok": False, "error": "Hardware not available (PCA9685 init failed)."}, 503

        try:
            pca.set_channel_angle_deg(int(ch), int(applied))
        except Exception as e:
            return False, {"ok": False, "error": f"Hardware command failed: {e}"}, 503
        _hw_angles[loc_key] = int(applied)
    else:
        _sim_angles[loc_key] = int(applied)

    return (
        True,
        {
            "ok": True,
            "mode": mode,
            "requested_angle": int(requested),
            "travel_applied_angle": int(travel_applied),
            "applied_angle": int(applied),
            "clamped": bool(clamped),
            "clamp_reason": "collision" if clamped else None,
            "collision": collision_details,
        },
        200,
    )


def _collision_snapshot_for_state(state_angles: Dict[str, int]) -> Dict[str, Any]:
    left_collides, left_details = _predict_collision_for_side(_dyn_vars, "left", state_angles)
    right_collides, right_details = _predict_collision_for_side(_dyn_vars, "right", state_angles)
    return {
        "left": {"collides": bool(left_collides), "details": left_details},
        "right": {"collides": bool(right_collides), "details": right_details},
    }


# -----------------------------
# API
# -----------------------------
@app.get("/api/config")
def api_get_config():
    global _saved_cfg, _draft_cfg, _dyn_vars, _hw_angles, _sim_angles
    available = _compute_available_channels(_draft_cfg)
    collision_status = {
        "normal": _collision_snapshot_for_state(_hw_angles),
        "test": _collision_snapshot_for_state(_sim_angles),
    }
    return jsonify(
        {
            "locations": [{"key": loc.key, "label": loc.label} for loc in LOCATIONS],
            "location_order": _location_order,
            "draft": _draft_cfg,
            "available_channels": available,
            "save_enabled": _dirty(_draft_cfg, _saved_cfg),
            "reset_enabled": _dirty(_draft_cfg, _saved_cfg),
            "angle_range": {"min": ANGLE_MIN_DEG, "max": ANGLE_MAX_DEG},
            "channel_range": {"min": CHANNEL_MIN, "max": CHANNEL_MAX},
            "hardware_available": pca is not None,
            "hardware_error": _pca_error,
            "dynamic_limits": _dyn_vars,
            "hw_angles": _hw_angles,
            "sim_angles": _sim_angles,
            "collision_status": collision_status,
        }
    )


@app.get("/api/dynamic_limits")
def api_get_dynamic_limits():
    global _dyn_vars
    return jsonify({"ok": True, "dynamic_limits": _dyn_vars})


@app.post("/api/dynamic_limits")
def api_set_dynamic_limits():
    global _dyn_vars

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    dv = data.get("dynamic_limits", None)
    if not isinstance(dv, dict):
        return jsonify({"ok": False, "error": "dynamic_limits must be an object"}), 400

    # Start from current to avoid accidental missing keys wiping config
    merged = json.loads(json.dumps(_dyn_vars))

    # active_side
    active_side = str(dv.get("active_side", merged.get("active_side", "left"))).strip().lower()
    if active_side not in ("left", "right"):
        active_side = merged.get("active_side", "left")
    merged["active_side"] = active_side

    # globals
    if "search_step_deg" in dv:
        merged["search_step_deg"] = _clamp_int(_coerce_int(dv.get("search_step_deg"), merged["search_step_deg"]), 1, 45)
    if "search_max_iters" in dv:
        merged["search_max_iters"] = _clamp_int(_coerce_int(dv.get("search_max_iters"), merged["search_max_iters"]), 1, 5000)

    # sides
    for side in ("left", "right"):
        if side not in dv:
            continue
        side_in = dv.get(side)
        if not isinstance(side_in, dict):
            continue
        side_out = merged.get(side, _default_dyn_side())
        for k in list(_default_dyn_side().keys()):
            if k in side_in:
                side_out[k] = _coerce_float(side_in.get(k), side_out.get(k, 0.0))
        merged[side] = side_out

    for side in ("left", "right"):
        err = _validate_dyn_side_values(side, merged[side])
        if err is not None:
            return jsonify({"ok": False, "error": err}), 400

    merged["version"] = 1
    _save_dyn_vars(merged)
    _dyn_vars = merged
    return jsonify({"ok": True, "dynamic_limits": _dyn_vars})


@app.get("/api/stances")
def api_get_stances():
    _ensure_stances_dir()
    stances: List[Dict[str, str]] = []
    invalid: List[Dict[str, str]] = []

    for path in sorted(STANCES_DIR.glob("*.json")):
        name = path.stem
        ok, _, err = _load_stance(name)
        if ok:
            stances.append({"name": name, "file": path.name})
        else:
            invalid.append({"name": name, "error": err})

    return jsonify({"ok": True, "stances": stances, "invalid": invalid, "default_stance": DEFAULT_STANCE_NAME})


@app.post("/api/stance/activate")
def api_activate_stance():
    global _hw_angles, _sim_angles

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    stance_name = str(data.get("stance", DEFAULT_STANCE_NAME)).strip()
    mode = str(data.get("mode", "normal")).strip().lower()
    if mode not in ("normal", "test"):
        mode = "normal"

    ok, angles, err = _load_stance(stance_name)
    if not ok:
        return jsonify({"ok": False, "error": err}), 400

    sequence = [k for k in _location_order if k in angles]
    for k in _stance_location_keys():
        if k not in sequence:
            sequence.append(k)

    results: List[Dict[str, Any]] = []
    delay_s = max(0.0, float(STANCE_STEP_DELAY_MS) / 1000.0)

    for idx, loc_key in enumerate(sequence):
        cmd_ok, payload, status = _execute_servo_command(loc_key, int(angles[loc_key]), mode)
        results.append({"location": loc_key, **payload})
        if not cmd_ok:
            return (
                jsonify(
                    {
                        "ok": False,
                        "stance": stance_name,
                        "mode": mode,
                        "delay_ms": STANCE_STEP_DELAY_MS,
                        "completed_count": len(results),
                        "results": results,
                        "hw_angles": _hw_angles,
                        "sim_angles": _sim_angles,
                        "error": payload.get("error", "stance activation failed"),
                    }
                ),
                status,
            )

        if idx < len(sequence) - 1 and delay_s > 0:
            time.sleep(delay_s)

    return jsonify(
        {
            "ok": True,
            "stance": stance_name,
            "mode": mode,
            "delay_ms": STANCE_STEP_DELAY_MS,
            "completed_count": len(results),
            "results": results,
            "hw_angles": _hw_angles,
            "sim_angles": _sim_angles,
        }
    )


@app.get("/api/channel_notes")
def api_get_channel_notes():
    return jsonify({"ok": True, "notes": _load_channel_notes()})


@app.post("/api/channel_notes")
def api_set_channel_notes():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    notes = data.get("notes", "")
    if not isinstance(notes, str):
        return jsonify({"ok": False, "error": "notes must be a string"}), 400
    if len(notes) > CHANNEL_NOTES_MAX_CHARS:
        return jsonify({"ok": False, "error": f"notes too large (max {CHANNEL_NOTES_MAX_CHARS} chars)"}), 413
    _save_channel_notes(notes)
    return jsonify({"ok": True})


@app.post("/api/channel")
def api_set_channel():
    global _draft_cfg

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    loc_key = str(data.get("location", "")).strip()
    if loc_key not in _draft_cfg["locations"]:
        return jsonify({"ok": False, "error": f"Unknown location '{loc_key}'"}), 400

    channel = data.get("channel", None)
    if channel is not None:
        try:
            channel = int(channel)
        except Exception:
            return jsonify({"ok": False, "error": "channel must be an integer or null"}), 400
        if not (CHANNEL_MIN <= channel <= CHANNEL_MAX):
            return jsonify({"ok": False, "error": f"channel out of range {CHANNEL_MIN}..{CHANNEL_MAX}"}), 400

    current = _draft_cfg["locations"][loc_key]["channel"]

    if channel is not None and channel != current:
        available = set(_compute_available_channels(_draft_cfg))
        if channel not in available:
            return jsonify({"ok": False, "error": f"Channel {channel} is already assigned elsewhere"}), 409

    _draft_cfg["locations"][loc_key]["channel"] = channel

    return jsonify(
        {
            "ok": True,
            "draft": _draft_cfg,
            "available_channels": _compute_available_channels(_draft_cfg),
            "save_enabled": _dirty(_draft_cfg, _saved_cfg),
            "reset_enabled": _dirty(_draft_cfg, _saved_cfg),
        }
    )


@app.post("/api/limits")
def api_set_limits():
    global _draft_cfg

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    loc_key = str(data.get("location", "")).strip()
    if loc_key not in _draft_cfg["locations"]:
        return jsonify({"ok": False, "error": f"Unknown location '{loc_key}'"}), 400

    raw_min = data.get("deg_min")
    raw_max = data.get("deg_max")
    if raw_min is None or raw_max is None:
        return jsonify({"ok": False, "error": "deg_min/deg_max are required"}), 400

    try:
        deg_min = _clamp_int(int(raw_min), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        deg_max = _clamp_int(int(raw_max), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    except Exception:
        return jsonify({"ok": False, "error": "deg_min/deg_max must be integers"}), 400

    if deg_max < deg_min:
        deg_min, deg_max = deg_max, deg_min

    invert = bool(data.get("invert", False))

    _draft_cfg["locations"][loc_key]["limits"] = {"deg_min": deg_min, "deg_max": deg_max, "invert": invert}

    return jsonify(
        {
            "ok": True,
            "draft": _draft_cfg,
            "save_enabled": _dirty(_draft_cfg, _saved_cfg),
            "reset_enabled": _dirty(_draft_cfg, _saved_cfg),
        }
    )


@app.post("/api/reset")
def api_reset():
    global _draft_cfg, _saved_cfg
    _draft_cfg = json.loads(json.dumps(_saved_cfg))
    return jsonify(
        {
            "ok": True,
            "draft": _draft_cfg,
            "available_channels": _compute_available_channels(_draft_cfg),
            "save_enabled": False,
            "reset_enabled": False,
        }
    )


@app.post("/api/save")
def api_save():
    global _draft_cfg, _saved_cfg

    ok, msg = _validate_no_duplicate_channels(_draft_cfg)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 409

    _save_config(_draft_cfg)
    _saved_cfg = json.loads(json.dumps(_draft_cfg))

    return jsonify({"ok": True, "save_enabled": False, "reset_enabled": False})


@app.post("/api/command")
def api_command():
    """
    Request body:
      {
        "location": "<loc_key>",
        "angle_deg": <int>,
        "mode": "normal" | "test"   (optional; default "normal")
      }

    Behavior:
      - Always applies travel clamp/invert (ServoLimits)
      - Always applies collision clamp (capsule model + step search)
      - In test mode: does NOT send PWM, but updates sim state (walk)
      - In normal mode: sends PWM (requires PCA), updates hw state
      - Returns applied_angle; UI should update input field to it
    """
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    loc_key = str(data.get("location", "")).strip()

    raw_angle = data.get("angle_deg")
    if raw_angle is None:
        return jsonify({"ok": False, "error": "angle_deg is required"}), 400

    try:
        requested = int(raw_angle)
    except Exception:
        return jsonify({"ok": False, "error": "angle_deg must be an integer"}), 400

    mode = str(data.get("mode", "normal")).strip().lower()
    ok, payload, status = _execute_servo_command(loc_key, requested, mode)
    return jsonify(payload), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
