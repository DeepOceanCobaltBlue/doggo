from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from hardware.pca9685 import ServoLimits

from .collision import collision_snapshot_for_state, predict_collision_for_side
from .kinematics import clamp_angle, infer_side
from .types import ANGLE_MAX_DEG, ANGLE_MIN_DEG, CommandResult, LOCATION_KEYS


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def apply_angle_to_state(loc_key: str, angle: int, state_angles: Dict[str, int]) -> Dict[str, int]:
    nxt = dict(state_angles)
    nxt[loc_key] = int(angle)
    return nxt


def solve_closest_safe_angle_step_search(
    dv: Dict[str, Any],
    side: str,
    loc_key: str,
    requested_angle: int,
    current_angle: int,
    state_angles: Dict[str, int],
    servo_limits_by_location: Dict[str, ServoLimits],
    step_deg: int,
    max_iters: int,
    angle_lo: int = ANGLE_MIN_DEG,
    angle_hi: int = ANGLE_MAX_DEG,
) -> Tuple[int, bool, Optional[Dict[str, Any]]]:
    requested_angle = int(requested_angle)
    current_angle = int(current_angle)
    step_deg = max(1, int(step_deg))
    max_iters = max(1, int(max_iters))
    lo = int(min(angle_lo, angle_hi))
    hi = int(max(angle_lo, angle_hi))
    requested_angle = clamp_int(requested_angle, lo, hi)
    current_angle = clamp_int(current_angle, lo, hi)

    test_state = apply_angle_to_state(loc_key, requested_angle, state_angles)
    collides, details = predict_collision_for_side(dv, side, test_state, servo_limits_by_location)
    if not collides:
        return requested_angle, False, details

    best_details = details
    dir_to_current = -1 if current_angle < requested_angle else +1 if current_angle > requested_angle else 0

    for i in range(1, max_iters + 1):
        delta = i * step_deg
        if dir_to_current == 0:
            candidates = [requested_angle - delta, requested_angle + delta]
        else:
            candidates = [requested_angle + dir_to_current * delta, requested_angle - dir_to_current * delta]

        in_range = [c for c in candidates if lo <= c <= hi]
        if not in_range:
            if (requested_angle - delta) < lo and (requested_angle + delta) > hi:
                break
            continue

        for candidate in in_range:
            test_state = apply_angle_to_state(loc_key, int(candidate), state_angles)
            collides, det = predict_collision_for_side(dv, side, test_state, servo_limits_by_location)
            best_details = det
            if not collides:
                return int(candidate), True, det

    fallback = clamp_int(int(current_angle), lo, hi)
    test_state = apply_angle_to_state(loc_key, fallback, state_angles)
    _, det = predict_collision_for_side(dv, side, test_state, servo_limits_by_location)
    return fallback, True, det


@dataclass
class SimulationEngine:
    dynamic_limits: Dict[str, Any]
    servo_limits_by_location: Dict[str, ServoLimits]
    default_neutral: int = 135
    states: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def ensure_state(self, state_name: str) -> Dict[str, int]:
        if state_name not in self.states:
            self.states[state_name] = {k: int(self.default_neutral) for k in LOCATION_KEYS}
        return self.states[state_name]

    def set_state(self, state_name: str, angles: Dict[str, int]) -> None:
        state = self.ensure_state(state_name)
        for k in LOCATION_KEYS:
            if k in angles:
                state[k] = clamp_int(int(angles[k]), ANGLE_MIN_DEG, ANGLE_MAX_DEG)

    def get_state(self, state_name: str) -> Dict[str, int]:
        return dict(self.ensure_state(state_name))

    def set_dynamic_limits(self, dynamic_limits: Dict[str, Any]) -> None:
        self.dynamic_limits = dynamic_limits

    def set_servo_limits_by_location(self, servo_limits_by_location: Dict[str, ServoLimits]) -> None:
        self.servo_limits_by_location = dict(servo_limits_by_location)

    def apply_command(
        self,
        state_name: str,
        loc_key: str,
        requested_angle: int,
        search_step_deg: Optional[int] = None,
        search_max_iters: Optional[int] = None,
    ) -> CommandResult:
        if loc_key not in self.servo_limits_by_location:
            raise KeyError(f"Unknown location '{loc_key}'")

        state = self.ensure_state(state_name)
        requested = clamp_int(int(requested_angle), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        limits = self.servo_limits_by_location[loc_key]
        travel_applied = clamp_angle(requested, limits)

        current_angle = int(state.get(loc_key, self.default_neutral))
        side = infer_side(loc_key)

        step_deg = clamp_int(
            int(self.dynamic_limits.get("search_step_deg", 1) if search_step_deg is None else search_step_deg),
            1,
            45,
        )
        max_iters = clamp_int(
            int(self.dynamic_limits.get("search_max_iters", 300) if search_max_iters is None else search_max_iters),
            1,
            5000,
        )

        deg_min = clamp_int(int(limits.deg_min), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        deg_max = clamp_int(int(limits.deg_max), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        if deg_max < deg_min:
            deg_min, deg_max = deg_max, deg_min

        applied, clamped, details = solve_closest_safe_angle_step_search(
            dv=self.dynamic_limits,
            side=side,
            loc_key=loc_key,
            requested_angle=int(travel_applied),
            current_angle=int(current_angle),
            state_angles=state,
            servo_limits_by_location=self.servo_limits_by_location,
            step_deg=step_deg,
            max_iters=max_iters,
            angle_lo=deg_min,
            angle_hi=deg_max,
        )

        state[loc_key] = int(applied)

        return CommandResult(
            requested_angle=int(requested),
            travel_applied_angle=int(travel_applied),
            applied_angle=int(applied),
            clamped=bool(clamped),
            clamp_reason="collision" if clamped else None,
            collision=details,
        )

    def apply_frame(self, state_name: str, updates: Dict[str, int]) -> Dict[str, int]:
        state = self.ensure_state(state_name)
        for loc_key, requested in updates.items():
            self.apply_command(state_name=state_name, loc_key=loc_key, requested_angle=int(requested))
        return dict(state)

    def collision_snapshot(self, state_name: str) -> Dict[str, Any]:
        state = self.ensure_state(state_name)
        return collision_snapshot_for_state(
            dv=self.dynamic_limits,
            state_angles=state,
            servo_limits_by_location=self.servo_limits_by_location,
        )
