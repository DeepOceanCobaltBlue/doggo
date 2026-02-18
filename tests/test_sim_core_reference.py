from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from hardware.pca9685 import ServoLimits
from sim_core.calibration import apply_sim_calibration_to_physical, build_calibration_fit_cache
from sim_core.collision import collision_snapshot_for_state, predict_collision_for_side
from sim_core.engine import SimulationEngine, solve_closest_safe_angle_step_search
from sim_core.kinematics import angles_pack_for_side_from_state
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
        # Regression: previously the implementation only checked the closest-distance pair.
        cfg = json.loads(Path("config/config_file.json").read_text())
        dv = json.loads(Path("config/dynamic_limits.json").read_text())
        stance = json.loads(Path("stances/default.json").read_text())

        limits = {}
        for k, v in cfg["locations"].items():
            lim = v["limits"]
            limits[k] = ServoLimits(
                deg_min=int(lim["deg_min"]),
                deg_max=int(lim["deg_max"]),
                invert=bool(lim["invert"]),
            )

        state = dict(stance.get("angles", {}))
        state["rear_right_hip"] = 182
        collides, details = predict_collision_for_side(dv, "right", state, limits)
        self.assertTrue(collides)
        self.assertIsInstance(details, dict)
        self.assertEqual(details.get("pair"), "front_thigh vs rear_shin")

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
