from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from hardware.pca9685 import ServoLimits

from .math_utils import clamp_int
from .safety_pipeline import SafetyPipeline


@dataclass(frozen=True)
class CommandExecutionResult:
    ok: bool
    payload: Dict[str, Any]
    status: int
    updated_state: Dict[str, int]


class CommandRunner:
    def __init__(self, safety_pipeline: SafetyPipeline):
        self._safety_pipeline = safety_pipeline

    def execute(
        self,
        *,
        state_name: str,
        loc_key: str,
        requested_angle: int,
        output_target: str,
        dynamic_limits: Mapping[str, Any],
        servo_limits_by_location: Mapping[str, ServoLimits],
        state_by_name: Mapping[str, Mapping[str, int]],
        channel_by_location: Mapping[str, Optional[int]],
        hardware: Optional[Any],
    ) -> CommandExecutionResult:
        if loc_key not in servo_limits_by_location:
            return CommandExecutionResult(
                ok=False,
                payload={"ok": False, "error": f"Unknown location '{loc_key}'"},
                status=400,
                updated_state=dict(state_by_name.get(state_name, {})),
            )

        if output_target not in ("hardware", "sim"):
            output_target = "hardware"

        requested = clamp_int(int(requested_angle), 0, 270)

        self._safety_pipeline.set_dynamic_limits(dynamic_limits)
        self._safety_pipeline.set_servo_limits_by_location(servo_limits_by_location)
        for sn, angles in state_by_name.items():
            self._safety_pipeline.set_state(sn, angles)

        try:
            sim_out = self._safety_pipeline.apply_command(state_name=state_name, loc_key=loc_key, requested_angle=requested)
        except KeyError:
            return CommandExecutionResult(
                ok=False,
                payload={"ok": False, "error": f"Unknown location '{loc_key}'"},
                status=400,
                updated_state=dict(state_by_name.get(state_name, {})),
            )

        if output_target == "hardware":
            ch = channel_by_location.get(loc_key)
            if ch is None:
                self._safety_pipeline.set_state(state_name, state_by_name.get(state_name, {}))
                return CommandExecutionResult(
                    ok=False,
                    payload={"ok": False, "error": f"Location '{loc_key}' is unassigned"},
                    status=409,
                    updated_state=dict(state_by_name.get(state_name, {})),
                )

            if hardware is None:
                self._safety_pipeline.set_state(state_name, state_by_name.get(state_name, {}))
                return CommandExecutionResult(
                    ok=False,
                    payload={"ok": False, "error": "Hardware not available (PCA9685 init failed)."},
                    status=503,
                    updated_state=dict(state_by_name.get(state_name, {})),
                )

            limits = servo_limits_by_location[loc_key]
            try:
                hardware.set_channel_angle_deg(int(ch), int(sim_out.applied_angle), limits=limits)
            except Exception as e:
                self._safety_pipeline.set_state(state_name, state_by_name.get(state_name, {}))
                return CommandExecutionResult(
                    ok=False,
                    payload={"ok": False, "error": f"Hardware command failed: {e}"},
                    status=503,
                    updated_state=dict(state_by_name.get(state_name, {})),
                )

        updated = self._safety_pipeline.get_state(state_name)

        return CommandExecutionResult(
            ok=True,
            payload={
                "ok": True,
                "requested_angle": int(sim_out.requested_angle),
                "travel_applied_angle": int(sim_out.travel_applied_angle),
                "applied_angle": int(sim_out.applied_angle),
                "clamped": bool(sim_out.clamped),
                "clamp_reason": sim_out.clamp_reason,
                "collision": sim_out.collision,
            },
            status=200,
            updated_state=updated,
        )
