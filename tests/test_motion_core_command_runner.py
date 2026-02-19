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
from motion_core import CommandRunner, SafetyPipeline
from sim_core.sim_store import default_dyn_vars
from sim_core.types import LOCATION_KEYS


def _limits() -> dict[str, ServoLimits]:
    return {k: ServoLimits(deg_min=0, deg_max=270, invert=False) for k in LOCATION_KEYS}


class _StubPCA:
    def __init__(self) -> None:
        self.calls = []

    def set_channel_angle_deg(self, channel, angle_deg, limits=None):
        self.calls.append((int(channel), int(angle_deg), limits))


class MotionCoreCommandRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        self.dynamic_limits = json.loads(json.dumps(dv))
        self.servo_limits = _limits()
        self.pipeline = SafetyPipeline(dynamic_limits=self.dynamic_limits, servo_limits_by_location=self.servo_limits)
        self.runner = CommandRunner(self.pipeline)
        self.states = {
            "normal": {k: 135 for k in LOCATION_KEYS},
            "test": {k: 135 for k in LOCATION_KEYS},
        }
        self.channels = {k: None for k in LOCATION_KEYS}
        self.channels["front_left_hip"] = 0

    def test_hardware_target_requires_channel(self) -> None:
        self.channels["front_left_hip"] = None
        out = self.runner.execute(
            state_name="normal",
            loc_key="front_left_hip",
            requested_angle=180,
            output_target="hardware",
            dynamic_limits=self.dynamic_limits,
            servo_limits_by_location=self.servo_limits,
            state_by_name=self.states,
            channel_by_location=self.channels,
            hardware=_StubPCA(),
        )
        self.assertFalse(out.ok)
        self.assertEqual(out.status, 409)

    def test_hardware_target_requires_device(self) -> None:
        out = self.runner.execute(
            state_name="normal",
            loc_key="front_left_hip",
            requested_angle=180,
            output_target="hardware",
            dynamic_limits=self.dynamic_limits,
            servo_limits_by_location=self.servo_limits,
            state_by_name=self.states,
            channel_by_location=self.channels,
            hardware=None,
        )
        self.assertFalse(out.ok)
        self.assertEqual(out.status, 503)

    def test_sim_target_collision_clamps(self) -> None:
        dv = json.loads(json.dumps(self.dynamic_limits))
        dv["left"]["front_thigh_radius_mm"] = 1000.0
        dv["left"]["front_shin_radius_mm"] = 1000.0
        dv["left"]["rear_thigh_radius_mm"] = 1000.0
        dv["left"]["rear_shin_radius_mm"] = 1000.0

        out = self.runner.execute(
            state_name="test",
            loc_key="front_left_hip",
            requested_angle=220,
            output_target="sim",
            dynamic_limits=dv,
            servo_limits_by_location=self.servo_limits,
            state_by_name=self.states,
            channel_by_location=self.channels,
            hardware=None,
        )
        self.assertTrue(out.ok)
        self.assertTrue(out.payload["clamped"])
        self.assertEqual(out.payload["clamp_reason"], "collision")


if __name__ == "__main__":
    unittest.main()
