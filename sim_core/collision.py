from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from hardware.pca9685 import ServoLimits

from .calibration import build_calibration_fit_cache
from .kinematics import (
    angles_pack_for_side_from_state,
    build_leg_capsules_for_side,
    dist,
    dot,
    norm2,
    pt_add,
    pt_mul,
    pt_sub,
)
from .types import Capsule


def dist_point_to_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ab = pt_sub(b, a)
    ap = pt_sub(p, a)
    denom = norm2(ab)
    if denom <= 1e-12:
        return dist(p, a)
    t = dot(ap, ab) / denom
    t = max(0.0, min(1.0, t))
    proj = pt_add(a, pt_mul(ab, t))
    return dist(p, proj)


def segments_intersect(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]) -> bool:
    def orient(p, q, r) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r) -> bool:
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    if abs(o1) <= 1e-9 and on_seg(a, c, b):
        return True
    if abs(o2) <= 1e-9 and on_seg(a, d, b):
        return True
    if abs(o3) <= 1e-9 and on_seg(c, a, d):
        return True
    if abs(o4) <= 1e-9 and on_seg(c, b, d):
        return True

    return False


def dist_segment_to_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], d: Tuple[float, float]) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        dist_point_to_segment(a, c, d),
        dist_point_to_segment(b, c, d),
        dist_point_to_segment(c, a, b),
        dist_point_to_segment(d, a, b),
    )


def capsules_min_distance(c1: Capsule, c2: Capsule) -> float:
    return dist_segment_to_segment(c1.a, c1.b, c2.a, c2.b)


def predict_collision_for_side(
    dv: Dict[str, Any],
    side: str,
    state_angles: Dict[str, int],
    servo_limits_by_location: Dict[str, ServoLimits],
    fit_cache: Optional[Dict[str, Optional[Tuple[float, float, str]]]] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    cache = fit_cache if fit_cache is not None else build_calibration_fit_cache(dv)
    pack = angles_pack_for_side_from_state(
        side=side,
        state_angles=state_angles,
        dv=dv,
        servo_limits_by_location=servo_limits_by_location,
        fit_cache=cache,
    )
    front_caps, rear_caps = build_leg_capsules_for_side(dv, side, pack)

    best = None
    for c1 in front_caps:
        for c2 in rear_caps:
            dmin = capsules_min_distance(c1, c2)
            thresh = float(c1.r + c2.r)
            margin = float(dmin - thresh)
            # Select the most critical pair by minimum clearance margin.
            # This catches overlaps even if another pair has smaller absolute distance.
            if best is None or margin < best[0]:
                best = (margin, dmin, thresh, c1.name, c2.name)

    if best is None:
        return False, None

    margin, dmin, thresh, n1, n2 = best
    details = {
        "pair": f"{n1} vs {n2}",
        "min_distance_mm": dmin,
        "threshold_mm": thresh,
    }
    return margin <= 0.0, details


def collision_snapshot_for_state(
    dv: Dict[str, Any],
    state_angles: Dict[str, int],
    servo_limits_by_location: Dict[str, ServoLimits],
) -> Dict[str, Any]:
    fit_cache = build_calibration_fit_cache(dv)
    left_collides, left_details = predict_collision_for_side(
        dv, "left", state_angles, servo_limits_by_location, fit_cache=fit_cache
    )
    right_collides, right_details = predict_collision_for_side(
        dv, "right", state_angles, servo_limits_by_location, fit_cache=fit_cache
    )
    return {
        "left": {"collides": bool(left_collides), "details": left_details},
        "right": {"collides": bool(right_collides), "details": right_details},
    }
