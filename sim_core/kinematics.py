from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from hardware.pca9685 import ServoLimits, resolve_logical_and_physical_angle

from .calibration import apply_sim_calibration_to_physical, build_calibration_fit_cache
from .types import ANGLE_MAX_DEG, ANGLE_MIN_DEG, Capsule


def infer_side(loc_key: str) -> str:
    if "_left_" in loc_key or loc_key.endswith("_left") or "left" in loc_key:
        return "left"
    if "_right_" in loc_key or loc_key.endswith("_right") or "right" in loc_key:
        return "right"
    return "left"


def normalize_deg(deg: float) -> float:
    x = deg % 360.0
    if x < 0:
        x += 360.0
    return x


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def unit_from_angle_deg(deg: float) -> Tuple[float, float]:
    r = deg_to_rad(deg)
    return (-math.cos(r), -math.sin(r))


def unit_from_knee_angle_deg(deg: float) -> Tuple[float, float]:
    r = deg_to_rad(deg)
    return (-math.cos(r), math.sin(r))


def rotate_ccw_deg(v: Tuple[float, float], deg: float) -> Tuple[float, float]:
    r = deg_to_rad(deg)
    c = math.cos(r)
    s = math.sin(r)
    x, y = v
    return (x * c - y * s, x * s + y * c)


def pt_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def pt_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def pt_mul(a: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (a[0] * s, a[1] * s)


def dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm2(a: Tuple[float, float]) -> float:
    return dot(a, a)


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def build_leg_capsules_for_side(
    dv: Dict[str, Any],
    side: str,
    angles: Dict[str, float],
) -> Tuple[List[Capsule], List[Capsule]]:
    s = dv[side]
    hip_dx = float(s["hip_dx_mm"])
    hip_y = float(s["hip_y_mm"])

    hf = (0.0, hip_y)
    hr = (hip_dx, hip_y) if side == "left" else (-hip_dx, hip_y)

    ltf = float(s["thigh_len_front_mm"])
    lsf = float(s["shin_len_front_mm"])
    ltr = float(s["thigh_len_rear_mm"])
    lsr = float(s["shin_len_rear_mm"])
    backoff_front = max(0.0, min(ltf, float(s.get("front_knee_attach_backoff_mm", 0.0))))
    backoff_rear = max(0.0, min(ltr, float(s.get("rear_knee_attach_backoff_mm", 0.0))))

    r_f_thigh = float(s["front_thigh_radius_mm"])
    r_f_shin = float(s["front_shin_radius_mm"])
    r_r_thigh = float(s["rear_thigh_radius_mm"])
    r_r_shin = float(s["rear_shin_radius_mm"])

    off_f_hip = float(s["front_hip_offset_deg"])
    off_f_knee = float(s["front_knee_offset_deg"])
    off_r_hip = float(s["rear_hip_offset_deg"])
    off_r_knee = float(s["rear_knee_offset_deg"])

    a_f_hip_cmd = float(angles.get("front_hip", 135.0))
    a_f_knee_cmd = float(angles.get("front_knee", 135.0))
    a_r_hip_cmd = float(angles.get("rear_hip", 135.0))
    a_r_knee_cmd = float(angles.get("rear_knee", 135.0))

    a_f_hip = normalize_deg(a_f_hip_cmd + off_f_hip)
    a_f_knee = normalize_deg(a_f_knee_cmd + off_f_knee)
    a_r_hip = normalize_deg(a_r_hip_cmd + off_r_hip)
    a_r_knee = normalize_deg(a_r_knee_cmd + off_r_knee)

    uf = unit_from_angle_deg(a_f_hip)
    kf_tip = pt_add(hf, pt_mul(uf, ltf))
    kf = pt_add(hf, pt_mul(uf, ltf - backoff_front))
    vf_local = unit_from_knee_angle_deg(a_f_knee)
    vf = rotate_ccw_deg(vf_local, a_f_hip)
    ff = pt_add(kf, pt_mul(vf, lsf))

    ur = unit_from_angle_deg(a_r_hip)
    kr_tip = pt_add(hr, pt_mul(ur, ltr))
    kr = pt_add(hr, pt_mul(ur, ltr - backoff_rear))
    vr_local = unit_from_knee_angle_deg(a_r_knee)
    vr = rotate_ccw_deg(vr_local, a_r_hip)
    fr = pt_add(kr, pt_mul(vr, lsr))

    front_caps = [
        Capsule(a=hf, b=kf_tip, r=r_f_thigh, name="front_thigh"),
        Capsule(a=kf, b=ff, r=r_f_shin, name="front_shin"),
    ]
    rear_caps = [
        Capsule(a=hr, b=kr_tip, r=r_r_thigh, name="rear_thigh"),
        Capsule(a=kr, b=fr, r=r_r_shin, name="rear_shin"),
    ]
    return front_caps, rear_caps


def angle_from_hip_unit(u: Tuple[float, float]) -> float:
    return normalize_deg(math.degrees(math.atan2(-u[1], -u[0])))


def angle_from_knee_local_unit(v_local: Tuple[float, float]) -> float:
    return normalize_deg(math.degrees(math.atan2(v_local[1], -v_local[0])))


def physical_sim_angle_from_state(
    loc_key: str,
    state_angles: Dict[str, int],
    servo_limits_by_location: Dict[str, ServoLimits],
) -> float:
    raw = int(state_angles.get(loc_key, 135))
    limits = servo_limits_by_location[loc_key]
    _, physical = resolve_logical_and_physical_angle(raw, limits)
    return float(physical)


def angles_pack_for_side_from_state(
    side: str,
    state_angles: Dict[str, int],
    dv: Dict[str, Any],
    servo_limits_by_location: Dict[str, ServoLimits],
    fit_cache: Optional[Dict[str, Optional[Tuple[float, float, str]]]] = None,
) -> Dict[str, float]:
    cache = fit_cache if fit_cache is not None else build_calibration_fit_cache(dv)
    side_vars = dv[side]

    if side == "left":
        keys = {
            "front_hip": "front_left_hip",
            "front_knee": "front_left_knee",
            "rear_hip": "rear_left_hip",
            "rear_knee": "rear_left_knee",
        }
    else:
        keys = {
            "front_hip": "front_right_hip",
            "front_knee": "front_right_knee",
            "rear_hip": "rear_right_hip",
            "rear_knee": "rear_right_knee",
        }

    phys = {k: physical_sim_angle_from_state(loc, state_angles, servo_limits_by_location) for k, loc in keys.items()}
    cal: Dict[str, Tuple[str, float]] = {}
    for k, loc in keys.items():
        cal[k] = apply_sim_calibration_to_physical(loc, phys[k], dv, fit_cache=cache)

    off_fh = float(side_vars["front_hip_offset_deg"])
    off_fk = float(side_vars["front_knee_offset_deg"])
    off_rh = float(side_vars["rear_hip_offset_deg"])
    off_rk = float(side_vars["rear_knee_offset_deg"])

    hip_line = (1.0, 0.0) if side == "left" else (-1.0, 0.0)

    out_front_hip = float(cal["front_hip"][1])
    if cal["front_hip"][0] == "hip_line_relative_deg":
        rel = normalize_deg(float(cal["front_hip"][1]) + off_fh)
        abs_fh = angle_from_hip_unit(rotate_ccw_deg(hip_line, rel))
        out_front_hip = normalize_deg(abs_fh - off_fh)

    out_rear_hip = float(cal["rear_hip"][1])
    if cal["rear_hip"][0] == "hip_line_relative_deg":
        rel = normalize_deg(float(cal["rear_hip"][1]) + off_rh)
        abs_rh = angle_from_hip_unit(rotate_ccw_deg(hip_line, rel))
        out_rear_hip = normalize_deg(abs_rh - off_rh)

    a_fh = normalize_deg(out_front_hip + off_fh)
    a_rh = normalize_deg(out_rear_hip + off_rh)
    uf = unit_from_angle_deg(a_fh)
    ur = unit_from_angle_deg(a_rh)

    out_front_knee = float(cal["front_knee"][1])
    if cal["front_knee"][0] == "knee_relative_deg":
        rel = normalize_deg(float(cal["front_knee"][1]) + off_fk)
        ba = (-uf[0], -uf[1])
        v_world = rotate_ccw_deg(ba, rel)
        v_local = rotate_ccw_deg(v_world, -a_fh)
        abs_fk = angle_from_knee_local_unit(v_local)
        out_front_knee = normalize_deg(abs_fk - off_fk)

    out_rear_knee = float(cal["rear_knee"][1])
    if cal["rear_knee"][0] == "knee_relative_deg":
        rel = normalize_deg(float(cal["rear_knee"][1]) + off_rk)
        ba = (-ur[0], -ur[1])
        v_world = rotate_ccw_deg(ba, rel)
        v_local = rotate_ccw_deg(v_world, -a_rh)
        abs_rk = angle_from_knee_local_unit(v_local)
        out_rear_knee = normalize_deg(abs_rk - off_rk)

    return {
        "front_hip": float(out_front_hip),
        "front_knee": float(out_front_knee),
        "rear_hip": float(out_rear_hip),
        "rear_knee": float(out_rear_knee),
    }


def clamp_angle(angle: int, limits: ServoLimits) -> int:
    deg_min = max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, int(limits.deg_min)))
    deg_max = max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, int(limits.deg_max)))
    if deg_max < deg_min:
        deg_min, deg_max = deg_max, deg_min
    return max(deg_min, min(deg_max, int(angle)))
