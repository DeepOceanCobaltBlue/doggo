from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .types import ANGLE_MAX_DEG, ANGLE_MIN_DEG


def linear_fit_from_pairs(pairs: List[Dict[str, float]]) -> Tuple[Optional[float], Optional[float]]:
    if len(pairs) < 2:
        return None, None
    xs = [float(p.get("commanded_deg", 0.0)) for p in pairs]
    ys = [float(p.get("actual_deg", 0.0)) for p in pairs]
    n = float(len(xs))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    var_x = sum((x - x_mean) ** 2 for x in xs)
    if var_x <= 1e-12:
        return None, None
    cov_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    m = cov_xy / var_x
    b = y_mean - m * x_mean
    return float(m), float(b)


def build_calibration_fit_cache(dv: Dict[str, Any]) -> Dict[str, Optional[Tuple[float, float, str]]]:
    out: Dict[str, Optional[Tuple[float, float, str]]] = {}
    profiles = dv.get("calibration_profiles", {})
    if not isinstance(profiles, dict):
        return out
    for name, prof in profiles.items():
        if not isinstance(prof, dict):
            out[str(name)] = None
            continue
        mode = str(prof.get("measurement_mode", "servo_physical_deg")).strip().lower()
        if mode not in ("servo_physical_deg", "hip_line_relative_deg", "knee_relative_deg"):
            mode = "servo_physical_deg"
        pairs = prof.get("pairs", [])
        if not isinstance(pairs, list):
            out[str(name)] = None
            continue
        m, b = linear_fit_from_pairs([p for p in pairs if isinstance(p, dict)])
        out[str(name)] = (m, b, mode) if m is not None and b is not None else None
    return out


def apply_sim_calibration_to_physical(
    loc_key: str,
    physical_deg: float,
    dv: Dict[str, Any],
    fit_cache: Optional[Dict[str, Optional[Tuple[float, float, str]]]] = None,
) -> Tuple[str, float]:
    profiles = dv.get("calibration_profiles", {})
    if not isinstance(profiles, dict):
        return "servo_physical_deg", float(physical_deg)

    jmap = dv.get("joint_calibration_map", {})
    if not isinstance(jmap, dict):
        jmap = {}
    prof_name = str(jmap.get(loc_key, "identity"))
    if prof_name not in profiles:
        prof_name = "identity"
    if prof_name not in profiles:
        return "servo_physical_deg", float(physical_deg)

    fit = None
    if fit_cache is not None:
        fit = fit_cache.get(prof_name)
    else:
        prof = profiles.get(prof_name, {})
        if isinstance(prof, dict):
            mode = str(prof.get("measurement_mode", "servo_physical_deg")).strip().lower()
            if mode not in ("servo_physical_deg", "hip_line_relative_deg", "knee_relative_deg"):
                mode = "servo_physical_deg"
            pairs = prof.get("pairs", [])
            if isinstance(pairs, list):
                m, b = linear_fit_from_pairs([p for p in pairs if isinstance(p, dict)])
                if m is not None and b is not None:
                    fit = (m, b, mode)

    if fit is None:
        return "servo_physical_deg", float(physical_deg)

    m, b, mode = fit
    predicted = float(m * float(physical_deg) + b)
    return mode, float(max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, predicted)))
