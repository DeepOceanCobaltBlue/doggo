from __future__ import annotations

import unittest

from motion_core import compile_timeline, normalize_and_validate_program_spec, packetize_timeline


class MotionCorePacketizerTests(unittest.TestCase):
    def test_packetize_timeline_sparse_and_dense(self) -> None:
        ok, spec, _msg = normalize_and_validate_program_spec(
            {
                "program_id": "pkt",
                "tick_ms": 20,
                "steps": [
                    {
                        "commands": [
                            {"location": "rear_left_hip", "target_angle": 150, "duration_ms": 40},
                        ]
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
            sparse_targets=False,
        )

        sparse_packets = packetize_timeline(timeline, sparse_targets=True, include_initial=False)
        dense_packets = packetize_timeline(timeline, sparse_targets=False, include_initial=False)

        self.assertEqual(len(sparse_packets), len(dense_packets))
        self.assertTrue(all("rear_left_hip" in p.targets for p in sparse_packets))
        self.assertTrue(all(set(p.targets.keys()) == {"rear_left_hip", "rear_right_hip"} for p in dense_packets))


if __name__ == "__main__":
    unittest.main()
