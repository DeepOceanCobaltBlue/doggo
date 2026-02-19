from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from hardware.pca9685 import ServoLimits
from sim_core.collision import collision_snapshot_for_state
from sim_core.engine import SimulationEngine
from sim_core.types import ANGLE_MAX_DEG, ANGLE_MIN_DEG, CommandResult
from .math_utils import clamp_int


@dataclass
class SafetyPipeline:
    """
    Shared movement safety pipeline wrapper around SimulationEngine.

    The caller owns mode/state naming (for example config_app's "normal" and "test").
    """

    dynamic_limits: Dict[str, Any]
    servo_limits_by_location: Dict[str, ServoLimits]
    default_neutral: int = 135

    def __post_init__(self) -> None:
        self._engine = SimulationEngine(
            dynamic_limits=self.dynamic_limits,
            servo_limits_by_location=self.servo_limits_by_location,
            default_neutral=int(self.default_neutral),
        )

    def set_dynamic_limits(self, dynamic_limits: Mapping[str, Any]) -> None:
        self.dynamic_limits = dict(dynamic_limits)
        self._engine.set_dynamic_limits(self.dynamic_limits)

    def set_servo_limits_by_location(self, servo_limits_by_location: Mapping[str, ServoLimits]) -> None:
        self.servo_limits_by_location = dict(servo_limits_by_location)
        self._engine.set_servo_limits_by_location(self.servo_limits_by_location)

    def set_state(self, state_name: str, angles: Mapping[str, int]) -> None:
        safe_angles = {
            str(k): clamp_int(int(v), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
            for k, v in angles.items()
        }
        self._engine.set_state(state_name, safe_angles)

    def get_state(self, state_name: str) -> Dict[str, int]:
        return self._engine.get_state(state_name)

    def apply_command(self, state_name: str, loc_key: str, requested_angle: int) -> CommandResult:
        requested = clamp_int(int(requested_angle), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        return self._engine.apply_command(
            state_name=state_name,
            loc_key=loc_key,
            requested_angle=requested,
        )

    def collision_snapshot_for_state(self, state_angles: Mapping[str, int]) -> Dict[str, Any]:
        return collision_snapshot_for_state(
            dv=self.dynamic_limits,
            state_angles={str(k): int(v) for k, v in state_angles.items()},
            servo_limits_by_location=self.servo_limits_by_location,
        )
