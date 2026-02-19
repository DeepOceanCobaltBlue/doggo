from __future__ import annotations

import unittest

from motion_core import (
    apply_command_slew_limits,
    apply_global_max_delta_per_tick,
    compile_timeline,
    normalize_and_validate_program_spec,
)


class MotionCoreGaitMutatorsTests(unittest.TestCase):
    def test_global_max_delta_per_tick_limits_tick_jump(self) -> None:
        ok, spec, _msg = normalize_and_validate_program_spec(
            {
                "program_id": "jump",
                "tick_ms": 20,
                "steps": [
                    {
                        "commands": [
                            {"location": "rear_left_hip", "target_angle": 230, "duration_ms": 20},
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
            sparse_targets=False,
            apply_slew_limits=False,
        )
        limited = apply_global_max_delta_per_tick(timeline, max_delta_per_tick=10)
        self.assertEqual(limited.ticks[1].targets["rear_left_hip"], 140)

    def test_command_slew_limits_respects_max_deg_per_sec(self) -> None:
        ok, spec, _msg = normalize_and_validate_program_spec(
            {
                "program_id": "slew",
                "tick_ms": 20,
                "steps": [
                    {
                        "commands": [
                            {
                                "location": "rear_left_hip",
                                "target_angle": 170,
                                "duration_ms": 100,
                                "max_deg_per_sec": 100.0,
                            },
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
            sparse_targets=False,
            apply_slew_limits=False,
        )
        slewed = apply_command_slew_limits(timeline, spec)

        # 100 deg/s at 20ms tick => budget 2 deg per tick.
        for i in range(1, len(slewed.ticks)):
            prev = slewed.ticks[i - 1].targets["rear_left_hip"]
            cur = slewed.ticks[i].targets["rear_left_hip"]
            self.assertLessEqual(abs(int(cur) - int(prev)), 2)


if __name__ == "__main__":
    unittest.main()
