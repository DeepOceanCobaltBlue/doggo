from __future__ import annotations

import unittest

from motion_core import validate_and_build_program_spec


class MotionCoreProgramModelsTests(unittest.TestCase):
    def test_valid_program_spec_builds(self) -> None:
        ok, spec, msg = validate_and_build_program_spec(
            {
                "program_id": "walk_test",
                "tick_ms": 20,
                "steps": [
                    {
                        "step_id": "lift",
                        "commands": [
                            {
                                "location": "front_left_hip",
                                "target_angle": 170,
                                "duration_ms": 300,
                                "easing": "ease_out",
                            },
                            {
                                "location": "front_left_knee",
                                "target_angle": 95,
                                "duration_ms": 300,
                            },
                        ],
                    }
                ],
            }
        )
        self.assertTrue(ok)
        self.assertEqual(msg, "ok")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.program_id, "walk_test")
        self.assertEqual(len(spec.steps), 1)
        self.assertEqual(len(spec.steps[0].commands), 2)

    def test_rejects_duplicate_locations_in_single_step(self) -> None:
        ok, spec, msg = validate_and_build_program_spec(
            {
                "program_id": "bad",
                "tick_ms": 20,
                "steps": [
                    {
                        "commands": [
                            {"location": "front_left_hip", "target_angle": 100, "duration_ms": 200},
                            {"location": "front_left_hip", "target_angle": 120, "duration_ms": 200},
                        ]
                    }
                ],
            }
        )
        self.assertFalse(ok)
        self.assertIsNone(spec)
        self.assertIn("duplicate location", msg)

    def test_rejects_out_of_range_angle(self) -> None:
        ok, spec, msg = validate_and_build_program_spec(
            {
                "program_id": "bad2",
                "tick_ms": 20,
                "steps": [
                    {
                        "commands": [
                            {"location": "front_left_hip", "target_angle": 400, "duration_ms": 200},
                        ]
                    }
                ],
            }
        )
        self.assertFalse(ok)
        self.assertIsNone(spec)
        self.assertIn("target_angle", msg)


if __name__ == "__main__":
    unittest.main()
