from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from .math_utils import clamp_int


def _normalize_deg(deg: float) -> float:
    x = float(deg) % 360.0
    if x < 0.0:
        x += 360.0
    return x


def _deg_to_rad(deg: float) -> float:
    return float(deg) * math.pi / 180.0


def _rad_to_deg(rad: float) -> float:
    return float(rad) * 180.0 / math.pi


def _unit_from_angle_deg(deg: float) -> Tuple[float, float]:
    r = _deg_to_rad(deg)
    return (-math.cos(r), -math.sin(r))


def _unit_from_knee_angle_deg(deg: float) -> Tuple[float, float]:
    r = _deg_to_rad(deg)
    return (-math.cos(r), math.sin(r))


def _rotate_ccw_deg(v: Tuple[float, float], deg: float) -> Tuple[float, float]:
    r = _deg_to_rad(deg)
    c = math.cos(r)
    s = math.sin(r)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def _angle_from_hip_unit(u: Tuple[float, float]) -> float:
    return _normalize_deg(_rad_to_deg(math.atan2(-u[1], -u[0])))


def _angle_from_knee_local_unit(v_local: Tuple[float, float]) -> float:
    return _normalize_deg(_rad_to_deg(math.atan2(v_local[1], -v_local[0])))


def _pt_add(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] + b[0], a[1] + b[1])


def _pt_sub(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return (a[0] - b[0], a[1] - b[1])


def _pt_mul(a: Tuple[float, float], s: float) -> Tuple[float, float]:
    return (a[0] * s, a[1] * s)


def _norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def _safe_acos(x: float) -> float:
    return math.acos(max(-1.0, min(1.0, float(x))))


def _clamp_to_loc_limits(loc_cfg: Optional[Dict[str, Any]], angle: float) -> int:
    raw = clamp_int(int(round(float(angle))), 0, 270)
    if not isinstance(loc_cfg, dict):
        return raw
    lim = loc_cfg.get("limits", None)
    if not isinstance(lim, dict):
        return raw
    lo = clamp_int(int(lim.get("deg_min", 0)), 0, 270)
    hi = clamp_int(int(lim.get("deg_max", 270)), 0, 270)
    if hi < lo:
        lo, hi = hi, lo
    return clamp_int(raw, lo, hi)


def leg_descriptor(side: str, leg: str, side_vars: Dict[str, Any]) -> Dict[str, float]:
    side_s = str(side).strip().lower()
    leg_s = str(leg).strip().lower()

    hip_y = float(side_vars["hip_y_mm"])
    hip_dx = float(side_vars["hip_dx_mm"])
    if leg_s == "front":
        hip = (0.0, hip_y)
        thigh_len = float(side_vars["thigh_len_front_mm"])
        shin_len = float(side_vars["shin_len_front_mm"])
        backoff = max(0.0, min(thigh_len, float(side_vars.get("front_knee_attach_backoff_mm", 0.0))))
        hip_offset = float(side_vars["front_hip_offset_deg"])
        knee_offset = float(side_vars["front_knee_offset_deg"])
    else:
        hip = (hip_dx, hip_y) if side_s == "left" else (-hip_dx, hip_y)
        thigh_len = float(side_vars["thigh_len_rear_mm"])
        shin_len = float(side_vars["shin_len_rear_mm"])
        backoff = max(0.0, min(thigh_len, float(side_vars.get("rear_knee_attach_backoff_mm", 0.0))))
        hip_offset = float(side_vars["rear_hip_offset_deg"])
        knee_offset = float(side_vars["rear_knee_offset_deg"])
    return {
        "hip_x": float(hip[0]),
        "hip_y": float(hip[1]),
        "a_mm": float(max(1e-6, thigh_len - backoff)),
        "b_mm": float(max(1e-6, shin_len)),
        "hip_offset_deg": float(hip_offset),
        "knee_offset_deg": float(knee_offset),
    }


def toe_from_leg_angles(desc: Dict[str, float], hip_cmd: float, knee_cmd: float) -> Tuple[float, float]:
    hip_abs = _normalize_deg(float(hip_cmd) + float(desc["hip_offset_deg"]))
    knee_abs = _normalize_deg(float(knee_cmd) + float(desc["knee_offset_deg"]))
    hip_pt = (float(desc["hip_x"]), float(desc["hip_y"]))
    a = float(desc["a_mm"])
    b = float(desc["b_mm"])

    u = _unit_from_angle_deg(hip_abs)
    k = _pt_add(hip_pt, _pt_mul(u, a))
    v_local = _unit_from_knee_angle_deg(knee_abs)
    v = _rotate_ccw_deg(v_local, hip_abs)
    toe = _pt_add(k, _pt_mul(v, b))
    return toe


def solve_leg_ik_for_toe(
    *,
    desc: Dict[str, float],
    toe_target: Tuple[float, float],
    prev_cmd: Optional[Tuple[float, float]] = None,
    hip_loc_cfg: Optional[Dict[str, Any]] = None,
    knee_loc_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hip_pt = (float(desc["hip_x"]), float(desc["hip_y"]))
    a = float(desc["a_mm"])
    b = float(desc["b_mm"])
    tx, ty = float(toe_target[0]), float(toe_target[1])
    r = _pt_sub((tx, ty), hip_pt)
    d = _norm(r)

    unreachable = False
    d_min = abs(a - b) + 1e-6
    d_max = (a + b) - 1e-6
    if d < d_min or d > d_max:
        unreachable = True
        if d <= 1e-9:
            r = (d_min, 0.0)
            d = d_min
        else:
            clamped_d = max(d_min, min(d_max, d))
            scale = clamped_d / d
            r = (r[0] * scale, r[1] * scale)
            d = clamped_d

    phi = math.atan2(r[1], r[0])
    cos_alpha = (a * a + d * d - b * b) / (2.0 * a * d)
    alpha = _safe_acos(cos_alpha)

    candidates: list[Tuple[float, float]] = []
    for sgn in (1.0, -1.0):
        theta_u = phi + (sgn * alpha)
        u = (math.cos(theta_u), math.sin(theta_u))
        k = _pt_add(hip_pt, _pt_mul(u, a))
        vk = _pt_sub((tx, ty), k)
        vn = _norm(vk)
        if vn <= 1e-9:
            continue
        v_world = (vk[0] / vn, vk[1] / vn)
        hip_abs = _angle_from_hip_unit(u)
        v_local = _rotate_ccw_deg(v_world, -hip_abs)
        knee_abs = _angle_from_knee_local_unit(v_local)
        hip_cmd = _normalize_deg(hip_abs - float(desc["hip_offset_deg"]))
        knee_cmd = _normalize_deg(knee_abs - float(desc["knee_offset_deg"]))
        candidates.append((hip_cmd, knee_cmd))

    if not candidates:
        # Fallback safe neutral when numeric edge case occurs.
        candidates = [(135.0, 135.0)]

    if prev_cmd is None:
        chosen = candidates[0]
    else:
        ph, pk = float(prev_cmd[0]), float(prev_cmd[1])

        def _score(c: Tuple[float, float]) -> float:
            return abs(float(c[0]) - ph) + abs(float(c[1]) - pk)

        chosen = min(candidates, key=_score)

    hip_raw = float(chosen[0])
    knee_raw = float(chosen[1])
    hip_cmd_i = _clamp_to_loc_limits(hip_loc_cfg, hip_raw)
    knee_cmd_i = _clamp_to_loc_limits(knee_loc_cfg, knee_raw)
    clamped = (hip_cmd_i != int(round(hip_raw))) or (knee_cmd_i != int(round(knee_raw)))
    toe_out = toe_from_leg_angles(desc, hip_cmd_i, knee_cmd_i)
    return {
        "hip_cmd": int(hip_cmd_i),
        "knee_cmd": int(knee_cmd_i),
        "toe_x": float(toe_out[0]),
        "toe_y": float(toe_out[1]),
        "unreachable": bool(unreachable),
        "clamped": bool(clamped),
    }
