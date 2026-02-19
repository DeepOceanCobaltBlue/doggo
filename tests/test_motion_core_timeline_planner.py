from __future__ import annotations

import unittest

from motion_core import compile_timeline, easing_value, normalize_and_validate_program_spec


class MotionCoreTimelinePlannerTests(unittest.TestCase):
    def test_easing_value_monotonic_endpoints(self) -> None:
        for mode in ("linear", "ease_in", "ease_out", "ease_in_out"):
            self.assertEqual(easing_value(mode, 0.0), 0.0)
            self.assertEqual(easing_value(mode, 1.0), 1.0)
            self.assertGreaterEqual(easing_value(mode, 0.75), easing_value(mode, 0.25))

    def test_compile_timeline_builds_shared_ticks_for_parallel_commands(self) -> None:
        ok, spec, msg = normalize_and_validate_program_spec(
            {
                "program_id": "walk_test",
                "tick_ms": 20,
                "steps": [
                    {
                        "step_id": "lift",
                        "commands": [
                            {"location": "rear_left_hip", "target_angle": 170, "duration_ms": 100, "easing": "linear"},
                            {"location": "rear_right_hip", "target_angle": 160, "duration_ms": 100, "easing": "linear"},
                        ],
                    }
                ],
            },
            location_keys=["rear_left_hip", "rear_right_hip"],
        )
        self.assertTrue(ok)
        self.assertEqual(msg, "ok")
        assert spec is not None

        timeline = compile_timeline(
            spec,
            location_keys=["rear_left_hip", "rear_right_hip"],
            start_state={"rear_left_hip": 130, "rear_right_hip": 130},
            sparse_targets=False,
        )

        self.assertEqual(timeline.tick_ms, 20)
        self.assertEqual(timeline.duration_ms, 100)
        self.assertEqual(len(timeline.ticks), 6)  # initial + 5 step ticks
        self.assertEqual(timeline.ticks[-1].targets["rear_left_hip"], 170)
        self.assertEqual(timeline.ticks[-1].targets["rear_right_hip"], 160)

    def test_compile_timeline_sparse_targets_only_emits_changes(self) -> None:
        ok, spec, _msg = normalize_and_validate_program_spec(
            {
                "program_id": "walk_test",
                "tick_ms": 20,
                "steps": [
                    {
                        "step_id": "single",
                        "commands": [
                            {"location": "rear_left_hip", "target_angle": 150, "duration_ms": 20, "easing": "linear"},
                        ],
                    }
                ],
            },
            location_keys=["rear_left_hip", "rear_right_hip"],
        )
        self.assertTrue(ok)
        assert spec is not None

        timeline = compile_timeline(
            spec,
            location_keys=["rear_left_hip", "rear_right_hip"],
            start_state={"rear_left_hip": 130, "rear_right_hip": 130},
            sparse_targets=True,
        )
        self.assertEqual(len(timeline.ticks), 2)
        self.assertEqual(set(timeline.ticks[1].targets.keys()), {"rear_left_hip"})
        self.assertEqual(timeline.ticks[1].targets["rear_left_hip"], 150)


if __name__ == "__main__":
    unittest.main()

