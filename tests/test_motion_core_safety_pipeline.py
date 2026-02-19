from __future__ import annotations

import json
import sys
import types
import unittest

if "smbus2" not in sys.modules:
    fake = types.ModuleType("smbus2")

    class _FakeSMBus:  # pragma: no cover - test bootstrap shim
        def __init__(self, *args, **kwargs):
            raise OSError("smbus unavailable in test environment")

    fake.SMBus = _FakeSMBus
    sys.modules["smbus2"] = fake

from hardware.pca9685 import ServoLimits
from motion_core import SafetyPipeline
from sim_core.sim_store import default_dyn_vars
from sim_core.types import LOCATION_KEYS


def _limits_with_inverted_right_knees() -> dict[str, ServoLimits]:
    limits = {k: ServoLimits(deg_min=0, deg_max=270, invert=False) for k in LOCATION_KEYS}
    limits["front_right_knee"] = ServoLimits(deg_min=0, deg_max=180, invert=True)
    limits["rear_right_knee"] = ServoLimits(deg_min=0, deg_max=180, invert=True)
    return limits


class MotionCoreSafetyPipelineTests(unittest.TestCase):
    def test_pipeline_applies_travel_clamp_before_collision_search(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        limits = _limits_with_inverted_right_knees()

        pipeline = SafetyPipeline(
            dynamic_limits=json.loads(json.dumps(dv)),
            servo_limits_by_location=limits,
        )
        pipeline.set_state("cfg_test", {k: 135 for k in LOCATION_KEYS})

        out = pipeline.apply_command("cfg_test", "front_right_knee", 250)
        self.assertEqual(out.requested_angle, 250)
        self.assertEqual(out.travel_applied_angle, 180)
        self.assertLessEqual(out.applied_angle, 180)

    def test_pipeline_collision_snapshot_and_clamp(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        dv["left"]["front_thigh_radius_mm"] = 1000.0
        dv["left"]["front_shin_radius_mm"] = 1000.0
        dv["left"]["rear_thigh_radius_mm"] = 1000.0
        dv["left"]["rear_shin_radius_mm"] = 1000.0

        pipeline = SafetyPipeline(
            dynamic_limits=json.loads(json.dumps(dv)),
            servo_limits_by_location=_limits_with_inverted_right_knees(),
        )

        state = {k: 135 for k in LOCATION_KEYS}
        pipeline.set_state("cfg_test", state)

        snap = pipeline.collision_snapshot_for_state(state)
        self.assertTrue(snap["left"]["collides"])

        out = pipeline.apply_command("cfg_test", "front_left_hip", 220)
        self.assertTrue(out.clamped)
        self.assertEqual(out.clamp_reason, "collision")
