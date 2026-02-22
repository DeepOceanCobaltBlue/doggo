from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hardware.pca9685 import ServoLimits
from sim_core.calibration import apply_sim_calibration_to_physical, build_calibration_fit_cache
from sim_core.collision import capsules_min_distance, collision_snapshot_for_state, predict_collision_for_side
from sim_core.engine import SimulationEngine, solve_closest_safe_angle_step_search
from sim_core.kinematics import angles_pack_for_side_from_state, build_leg_capsules_for_side
from sim_core.sim_store import default_dyn_vars, load_stance, save_dyn_vars
from sim_core.types import LOCATION_KEYS


def _default_servo_limits() -> dict[str, ServoLimits]:
    limits = {k: ServoLimits(deg_min=0, deg_max=270, invert=False) for k in LOCATION_KEYS}
    limits["front_right_knee"] = ServoLimits(deg_min=0, deg_max=270, invert=True)
    limits["rear_right_knee"] = ServoLimits(deg_min=0, deg_max=270, invert=True)
    return limits


class SimCoreReference(unittest.TestCase):
    def test_default_dyn_vars_include_expected_profiles(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        self.assertEqual(dv["version"], 2)
        self.assertIn("identity", dv["calibration_profiles"])
        self.assertIn("hip_standard", dv["calibration_profiles"])
        self.assertIn("knee_standard", dv["calibration_profiles"])

    def test_calibration_cache_and_apply(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        cache = build_calibration_fit_cache(dv)
        self.assertIn("identity", cache)

        mode, val = apply_sim_calibration_to_physical("front_left_hip", 150.0, dv, fit_cache=cache)
        self.assertIn(mode, ("servo_physical_deg", "hip_line_relative_deg", "knee_relative_deg"))
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 270.0)

    def test_angles_pack_respects_invert_for_physical_mapping(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        dv["joint_calibration_map"] = {loc_key: "identity" for loc_key in LOCATION_KEYS}
        servo_limits = _default_servo_limits()

        state = {
            "front_right_hip": 100,
            "front_right_knee": 180,
            "rear_right_hip": 120,
            "rear_right_knee": 170,
        }
        pack = angles_pack_for_side_from_state("right", state, dv=dv, servo_limits_by_location=servo_limits)
        self.assertEqual(pack["front_hip"], 100.0)
        self.assertEqual(pack["front_knee"], 90.0)
        self.assertEqual(pack["rear_hip"], 120.0)
        self.assertEqual(pack["rear_knee"], 100.0)

    def test_collision_predict_and_snapshot_shape(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        servo_limits = _default_servo_limits()
        state = {k: 135 for k in LOCATION_KEYS}

        collides, details = predict_collision_for_side(dv, "left", state, servo_limits)
        self.assertIsInstance(collides, bool)
        self.assertIsInstance(details, dict)
        self.assertIn("pair", details)

        snap = collision_snapshot_for_state(dv, state, servo_limits)
        self.assertIn("left", snap)
        self.assertIn("right", snap)

    def test_collision_flags_any_overlapping_pair_not_only_closest_distance_pair(self) -> None:
        # Regression target:
        # A pair can overlap by radius margin even when it's not the pair with minimum raw segment distance.
        # We search for such a configuration in a deterministic synthetic setup.
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        dv["right"]["front_thigh_radius_mm"] = 35.0
        dv["right"]["rear_shin_radius_mm"] = 35.0
        dv["right"]["front_shin_radius_mm"] = 1.0
        dv["right"]["rear_thigh_radius_mm"] = 1.0

        limits = _default_servo_limits()
        base_state = {k: 135 for k in LOCATION_KEYS}

        found_state = None
        for rear_hip in range(90, 231):
            state = dict(base_state)
            state["rear_right_hip"] = rear_hip

            pack = angles_pack_for_side_from_state("right", state, dv=dv, servo_limits_by_location=limits)
            front_caps, rear_caps = build_leg_capsules_for_side(dv, "right", pack)

            pairs = []
            for c1 in front_caps:
                for c2 in rear_caps:
                    dmin = capsules_min_distance(c1, c2)
                    thresh = float(c1.r + c2.r)
                    margin = float(dmin - thresh)
                    pairs.append((dmin, margin))

            by_distance = min(pairs, key=lambda x: x[0])
            any_overlap = any(margin <= 0.0 for _d, margin in pairs)
            if any_overlap and by_distance[1] > 0.0:
                found_state = state
                break

        self.assertIsNotNone(found_state, "failed to synthesize overlap-vs-closest-distance regression scenario")
        collides, details = predict_collision_for_side(dv, "right", found_state, limits)
        self.assertTrue(collides)
        self.assertIsInstance(details, dict)

    def test_knee_attach_backoff_moves_shin_start_without_changing_thigh_capsule(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        angles = {"front_hip": 140.0, "front_knee": 135.0, "rear_hip": 130.0, "rear_knee": 150.0}

        front0, _ = build_leg_capsules_for_side(dv, "left", angles)
        dv["left"]["front_knee_attach_backoff_mm"] = 10.0
        front1, _ = build_leg_capsules_for_side(dv, "left", angles)

        thigh0 = next(c for c in front0 if c.name == "front_thigh")
        shin0 = next(c for c in front0 if c.name == "front_shin")
        thigh1 = next(c for c in front1 if c.name == "front_thigh")
        shin1 = next(c for c in front1 if c.name == "front_shin")

        self.assertEqual(thigh0.a, thigh1.a)
        self.assertEqual(thigh0.b, thigh1.b)
        self.assertNotEqual(shin0.a, shin1.a)

    def test_step_search_returns_expected_tuple(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        dv["left"]["front_thigh_radius_mm"] = 1000.0
        dv["left"]["front_shin_radius_mm"] = 1000.0
        dv["left"]["rear_thigh_radius_mm"] = 1000.0
        dv["left"]["rear_shin_radius_mm"] = 1000.0

        state = {k: 135 for k in LOCATION_KEYS}
        servo_limits = _default_servo_limits()

        out = solve_closest_safe_angle_step_search(
            dv=dv,
            side="left",
            loc_key="front_left_hip",
            requested_angle=200,
            current_angle=135,
            state_angles=state,
            servo_limits_by_location=servo_limits,
            step_deg=1,
            max_iters=300,
            angle_lo=0,
            angle_hi=270,
        )
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[0], int)
        self.assertIsInstance(out[1], bool)

    def test_engine_apply_command_tracks_state(self) -> None:
        dv = default_dyn_vars(location_keys=LOCATION_KEYS)
        servo_limits = _default_servo_limits()

        engine = SimulationEngine(dynamic_limits=json.loads(json.dumps(dv)), servo_limits_by_location=servo_limits)
        engine.set_state("test", {k: 135 for k in LOCATION_KEYS})

        out = engine.apply_command("test", "front_left_hip", 180)
        self.assertIsInstance(out.requested_angle, int)
        self.assertIn("front_left_hip", engine.get_state("test"))
        self.assertEqual(engine.get_state("test")["front_left_hip"], out.applied_angle)

    def test_sim_store_save_and_load_stance(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            dyn_file = base / "dynamic_limits.json"
            stance_dir = base / "stances"
            stance_dir.mkdir(parents=True, exist_ok=True)

            dv = default_dyn_vars(location_keys=LOCATION_KEYS)
            save_dyn_vars(dyn_file, dv)
            self.assertTrue(dyn_file.exists())

            stance_payload = {"angles": {k: 135 for k in LOCATION_KEYS}}
            (stance_dir / "default.json").write_text(json.dumps(stance_payload))
            ok, angles, err = load_stance(stance_dir, "default", location_keys=LOCATION_KEYS)
            self.assertTrue(ok)
            self.assertEqual(err, "")
            self.assertEqual(angles["front_left_hip"], 135)


if __name__ == "__main__":
    unittest.main()
