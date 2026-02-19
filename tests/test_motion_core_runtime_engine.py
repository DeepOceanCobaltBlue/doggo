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
from motion_core import (
    CommandRunner,
    ProgramRuntimeEngine,
    SafetyPipeline,
    compile_timeline,
    normalize_and_validate_program_spec,
)
from sim_core.sim_store import default_dyn_vars
from sim_core.types import LOCATION_KEYS


class MotionCoreRuntimeEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        self.dynamic_limits = json.loads(json.dumps(dv))
        self.servo_limits = {k: ServoLimits(deg_min=0, deg_max=270, invert=False) for k in LOCATION_KEYS}
        self.pipeline = SafetyPipeline(dynamic_limits=self.dynamic_limits, servo_limits_by_location=self.servo_limits)
        self.runner = CommandRunner(self.pipeline)
        self.engine = ProgramRuntimeEngine(self.runner)

    def test_runtime_engine_executes_to_completion(self) -> None:
        ok, spec, _msg = normalize_and_validate_program_spec(
            {
                "program_id": "rt",
                "tick_ms": 20,
                "steps": [
                    {
                        "commands": [
                            {"location": "rear_left_hip", "target_angle": 150, "duration_ms": 40},
                        ]
                    }
                ],
            },
            location_keys=["rear_left_hip"],
        )
        self.assertTrue(ok)
        assert spec is not None

        timeline = compile_timeline(
            spec,
            location_keys=["rear_left_hip"],
            start_state={"rear_left_hip": 130},
            sparse_targets=True,
        )

        result = self.engine.run(
            timeline,
            state_name="test",
            output_target="sim",
            dynamic_limits=self.dynamic_limits,
            servo_limits_by_location=self.servo_limits,
            channel_by_location={k: None for k in LOCATION_KEYS},
            hardware=None,
            state_by_name={"test": {k: (130 if k == "rear_left_hip" else 135) for k in LOCATION_KEYS}},
            stop_on_clamp=False,
            realtime=False,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "completed")
        self.assertEqual(result.final_state_by_name["test"]["rear_left_hip"], 150)

    def test_runtime_engine_stop_on_clamp(self) -> None:
        dv = json.loads(json.dumps(self.dynamic_limits))
        dv["left"]["front_thigh_radius_mm"] = 1000.0
        dv["left"]["front_shin_radius_mm"] = 1000.0
        dv["left"]["rear_thigh_radius_mm"] = 1000.0
        dv["left"]["rear_shin_radius_mm"] = 1000.0

        ok, spec, _msg = normalize_and_validate_program_spec(
            {
                "program_id": "rt_clamp",
                "tick_ms": 20,
                "steps": [
                    {
                        "commands": [
                            {"location": "front_left_hip", "target_angle": 220, "duration_ms": 20},
                        ]
                    }
                ],
            },
            location_keys=["front_left_hip"],
        )
        self.assertTrue(ok)
        assert spec is not None

        timeline = compile_timeline(
            spec,
            location_keys=["front_left_hip"],
            start_state={"front_left_hip": 135},
            sparse_targets=True,
        )

        result = self.engine.run(
            timeline,
            state_name="test",
            output_target="sim",
            dynamic_limits=dv,
            servo_limits_by_location=self.servo_limits,
            channel_by_location={k: None for k in LOCATION_KEYS},
            hardware=None,
            state_by_name={"test": {k: 135 for k in LOCATION_KEYS}},
            stop_on_clamp=True,
            realtime=False,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "clamped")


if __name__ == "__main__":
    unittest.main()
