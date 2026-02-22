from __future__ import annotations

import json
import importlib
import math
import sys
import types
import unittest
from pathlib import Path

from motion_core import servo_limits_from_config_item
from sim_core import sim_store
from sim_core.kinematics import (
    angles_pack_for_side_from_state,
    build_leg_capsules_for_side,
    rotate_ccw_deg,
    unit_from_angle_deg,
    unit_from_knee_angle_deg,
)
from sim_core.types import LOCATION_KEYS

try:
    import flask  # noqa: F401
    HAVE_FLASK = True
except Exception:
    HAVE_FLASK = False


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config" / "config_file.json"
LOCATION_ORDER_FILE = ROOT / "config" / "location_order.json"


class NamingAndSchemaContracts(unittest.TestCase):
    def test_config_and_order_use_rear_keys(self) -> None:
        cfg = json.loads(CONFIG_FILE.read_text())
        order = json.loads(LOCATION_ORDER_FILE.read_text())

        loc_keys = set(cfg["locations"].keys())
        order_keys = set(order["position_order"])

        self.assertFalse(any("back_" in k for k in loc_keys))
        self.assertFalse(any("back_" in k for k in order_keys))
        self.assertEqual(loc_keys, order_keys)

@unittest.skipUnless(HAVE_FLASK, "Flask not installed in this environment")
class ConfigApiContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if "smbus2" not in sys.modules:
            fake = types.ModuleType("smbus2")

            class _FakeSMBus:  # pragma: no cover - test bootstrap shim
                def __init__(self, *args, **kwargs):
                    raise OSError("smbus unavailable in test environment")

            fake.SMBus = _FakeSMBus
            sys.modules["smbus2"] = fake

        cls.config_app = importlib.import_module("config_app")
        cls.pca_mod = importlib.import_module("hardware.pca9685")

    def test_config_api_exposes_dynamic_limits_key(self) -> None:
        client = self.config_app.app.test_client()
        resp = client.get("/api/config")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("dynamic_limits", data)
        self.assertIn("collision_status", data)
        self.assertNotIn("dynamic_limit_variables", data)

    def test_dynamic_limits_endpoint_contract(self) -> None:
        client = self.config_app.app.test_client()
        resp = client.post("/api/dynamic_limits", json={"dynamic_limits": {}})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("dynamic_limits", data)
        self.assertNotIn("dynamic_limit_variables", data)

    def test_stances_list_and_activate_default_test_mode(self) -> None:
        client = self.config_app.app.test_client()

        resp = client.get("/api/stances")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["ok"])
        names = [s["name"] for s in data["stances"]]
        self.assertIn("default", names)

        resp = client.post("/api/stance/activate", json={"stance": "default", "mode": "test"})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["stance"], "default")
        self.assertEqual(body["mode"], "test")
        self.assertEqual(body["completed_count"], 8)

    def test_activate_stance_missing_file_rejected(self) -> None:
        client = self.config_app.app.test_client()
        resp = client.post("/api/stance/activate", json={"stance": "does_not_exist", "mode": "test"})
        self.assertEqual(resp.status_code, 400)

    def test_dynamic_limits_reject_nonpositive_radii(self) -> None:
        client = self.config_app.app.test_client()
        resp = client.post(
            "/api/dynamic_limits",
            json={"dynamic_limits": {"left": {"front_thigh_radius_mm": 0}}},
        )
        self.assertEqual(resp.status_code, 400)

    def test_dynamic_limits_reject_nonpositive_lengths(self) -> None:
        client = self.config_app.app.test_client()
        resp = client.post(
            "/api/dynamic_limits",
            json={"dynamic_limits": {"left": {"thigh_len_front_mm": 0}}},
        )
        self.assertEqual(resp.status_code, 400)

    def test_dynamic_limits_reject_negative_knee_attach_backoff(self) -> None:
        client = self.config_app.app.test_client()
        resp = client.post(
            "/api/dynamic_limits",
            json={"dynamic_limits": {"left": {"front_knee_attach_backoff_mm": -1}}},
        )
        self.assertEqual(resp.status_code, 400)

    def test_dynamic_limits_reject_knee_attach_backoff_beyond_thigh_length(self) -> None:
        client = self.config_app.app.test_client()
        cur = client.get("/api/dynamic_limits").get_json()["dynamic_limits"]
        too_large = float(cur["left"]["thigh_len_front_mm"]) + 1.0
        resp = client.post(
            "/api/dynamic_limits",
            json={"dynamic_limits": {"left": {"front_knee_attach_backoff_mm": too_large}}},
        )
        self.assertEqual(resp.status_code, 400)

    def test_command_collision_clamps_when_geometry_forces_overlap(self) -> None:
        client = self.config_app.app.test_client()
        original = client.get("/api/dynamic_limits").get_json()["dynamic_limits"]

        try:
            # Force guaranteed overlap by making radii very large on the left side.
            resp = client.post(
                "/api/dynamic_limits",
                json={
                    "dynamic_limits": {
                        "left": {
                            "front_thigh_radius_mm": 1000.0,
                            "front_shin_radius_mm": 1000.0,
                            "rear_thigh_radius_mm": 1000.0,
                            "rear_shin_radius_mm": 1000.0,
                        }
                    }
                },
            )
            self.assertEqual(resp.status_code, 200)

            cmd = client.post(
                "/api/command",
                json={"location": "front_left_hip", "angle_deg": 200, "mode": "test"},
            )
            self.assertEqual(cmd.status_code, 200)
            data = cmd.get_json()
            self.assertTrue(data["clamped"])
            self.assertEqual(data["clamp_reason"], "collision")
        finally:
            restore = client.post("/api/dynamic_limits", json={"dynamic_limits": original})
            self.assertEqual(restore.status_code, 200)

    def test_invert_neutral_stays_stable_across_clamp_window_changes(self) -> None:
        client = self.config_app.app.test_client()
        cfg = client.get("/api/config").get_json()
        original_limits = dict(cfg["draft"]["locations"]["front_left_knee"]["limits"])
        loc = "front_left_knee"

        try:
            r1 = client.post(
                "/api/limits",
                json={"location": loc, "deg_min": 30, "deg_max": 180, "invert": True},
            )
            self.assertEqual(r1.status_code, 200)
            c1 = client.post("/api/command", json={"location": loc, "angle_deg": 135, "mode": "test"})
            self.assertEqual(c1.status_code, 200)
            d1 = c1.get_json()

            r2 = client.post(
                "/api/limits",
                json={"location": loc, "deg_min": 60, "deg_max": 210, "invert": True},
            )
            self.assertEqual(r2.status_code, 200)
            c2 = client.post("/api/command", json={"location": loc, "angle_deg": 135, "mode": "test"})
            self.assertEqual(c2.status_code, 200)
            d2 = c2.get_json()

            self.assertEqual(d1["travel_applied_angle"], 135)
            self.assertEqual(d2["travel_applied_angle"], 135)
            self.assertEqual(d1["travel_applied_angle"], d2["travel_applied_angle"])
        finally:
            restore = client.post(
                "/api/limits",
                json={
                    "location": loc,
                    "deg_min": original_limits["deg_min"],
                    "deg_max": original_limits["deg_max"],
                    "invert": original_limits["invert"],
                },
            )
            self.assertEqual(restore.status_code, 200)

    def test_invert_keeps_test_mode_travel_clamp_logical(self) -> None:
        client = self.config_app.app.test_client()
        # front_right_knee is invert=true with deg_max=180 in default config.
        cmd = client.post(
            "/api/command",
            json={"location": "front_right_knee", "angle_deg": 250, "mode": "test"},
        )
        self.assertEqual(cmd.status_code, 200)
        data = cmd.get_json()
        self.assertEqual(data["requested_angle"], 250)
        self.assertEqual(data["travel_applied_angle"], 180)

    def test_command_normal_rejects_unassigned_channel(self) -> None:
        client = self.config_app.app.test_client()
        module = self.config_app
        loc = "front_left_hip"
        original = module._draft_cfg["locations"][loc]["channel"]
        try:
            module._draft_cfg["locations"][loc]["channel"] = None
            resp = client.post("/api/command", json={"location": loc, "angle_deg": 150, "mode": "normal"})
            self.assertEqual(resp.status_code, 409)
            body = resp.get_json()
            self.assertIn("unassigned", str(body.get("error", "")).lower())
        finally:
            module._draft_cfg["locations"][loc]["channel"] = original

    def test_command_normal_rejects_when_hardware_unavailable(self) -> None:
        client = self.config_app.app.test_client()
        module = self.config_app
        loc = "front_left_hip"
        original_pca = module.pca
        original_channel = module._draft_cfg["locations"][loc]["channel"]
        try:
            module._draft_cfg["locations"][loc]["channel"] = 0
            module.pca = None
            resp = client.post("/api/command", json={"location": loc, "angle_deg": 150, "mode": "normal"})
            self.assertEqual(resp.status_code, 503)
        finally:
            module.pca = original_pca
            module._draft_cfg["locations"][loc]["channel"] = original_channel

    def test_command_mode_partition_test_only_updates_sim_state(self) -> None:
        client = self.config_app.app.test_client()
        before = client.get("/api/config").get_json()
        loc = "front_left_hip"
        hw_before = int(before["hw_angles"][loc])
        sim_before = int(before["sim_angles"][loc])
        target = min(270, sim_before + 10)

        resp = client.post("/api/command", json={"location": loc, "angle_deg": target, "mode": "test"})
        self.assertEqual(resp.status_code, 200)

        after = client.get("/api/config").get_json()
        self.assertEqual(int(after["hw_angles"][loc]), hw_before)
        self.assertEqual(int(after["sim_angles"][loc]), int(resp.get_json()["applied_angle"]))

    def test_command_mode_partition_normal_does_not_update_sim_state(self) -> None:
        client = self.config_app.app.test_client()
        module = self.config_app
        loc = "front_left_hip"

        class _StubPCA:
            def set_channel_angle_deg(self, channel, angle_deg, limits=None):
                return None

        original_pca = module.pca
        original_channel = module._draft_cfg["locations"][loc]["channel"]
        try:
            module.pca = _StubPCA()
            module._draft_cfg["locations"][loc]["channel"] = 0
            before = client.get("/api/config").get_json()
            sim_before = int(before["sim_angles"][loc])
            target = min(270, int(before["hw_angles"][loc]) + 10)

            resp = client.post("/api/command", json={"location": loc, "angle_deg": target, "mode": "normal"})
            self.assertEqual(resp.status_code, 200)

            after = client.get("/api/config").get_json()
            self.assertEqual(int(after["sim_angles"][loc]), sim_before)
        finally:
            module.pca = original_pca
            module._draft_cfg["locations"][loc]["channel"] = original_channel

    def test_normal_mode_delegates_invert_to_hardware_layer(self) -> None:
        client = self.config_app.app.test_client()
        module = self.config_app

        class _StubPCA:
            def __init__(self) -> None:
                self.calls = []

            def set_channel_angle_deg(self, channel, angle_deg, limits=None):  # pragma: no cover - assertion target
                self.calls.append((channel, angle_deg, limits))

        original_pca = module.pca
        stub = _StubPCA()
        module.pca = stub
        try:
            cmd = client.post(
                "/api/command",
                json={"location": "front_right_knee", "angle_deg": 250, "mode": "normal"},
            )
            self.assertEqual(cmd.status_code, 200)
            data = cmd.get_json()
            self.assertEqual(data["travel_applied_angle"], 180)
            self.assertEqual(len(stub.calls), 1)

            ch, angle_sent, limits = stub.calls[0]
            expected_ch = int(module._draft_cfg["locations"]["front_right_knee"]["channel"])
            self.assertEqual(ch, expected_ch)
            self.assertEqual(angle_sent, data["applied_angle"])
            self.assertIsNotNone(limits)
            expected_limits = module._draft_cfg["locations"]["front_right_knee"]["limits"]
            self.assertEqual(bool(limits.invert), bool(expected_limits["invert"]))
            self.assertEqual(int(limits.deg_min), int(expected_limits["deg_min"]))
            self.assertEqual(int(limits.deg_max), int(expected_limits["deg_max"]))
        finally:
            module.pca = original_pca

    def test_driver_resolve_logical_then_invert_output(self) -> None:
        limits_inv = self.pca_mod.ServoLimits(deg_min=30, deg_max=180, invert=True)
        logical, physical = self.pca_mod.resolve_logical_and_physical_angle(250, limits_inv)
        self.assertEqual(logical, 180)
        self.assertEqual(physical, 90)

        limits_noninv = self.pca_mod.ServoLimits(deg_min=30, deg_max=180, invert=False)
        logical2, physical2 = self.pca_mod.resolve_logical_and_physical_angle(250, limits_noninv)
        self.assertEqual(logical2, 180)
        self.assertEqual(physical2, 180)

    def test_sim_pack_uses_physical_angles_for_inverted_joints(self) -> None:
        module = self.config_app
        state = {
            "front_right_hip": 100,
            "front_right_knee": 180,  # invert=true in default config
            "rear_right_hip": 120,
            "rear_right_knee": 170,   # invert=true in default config
        }
        dv = sim_store.default_dyn_vars(location_keys=LOCATION_KEYS)
        dv["joint_calibration_map"] = {loc_key: "identity" for loc_key in LOCATION_KEYS}
        servo_limits = {
            loc_key: servo_limits_from_config_item(module._draft_cfg["locations"][loc_key])
            for loc_key in LOCATION_KEYS
        }
        pack = angles_pack_for_side_from_state("right", state, dv=dv, servo_limits_by_location=servo_limits)
        self.assertEqual(pack["front_hip"], 100.0)
        self.assertEqual(pack["front_knee"], 90.0)
        self.assertEqual(pack["rear_hip"], 120.0)
        self.assertEqual(pack["rear_knee"], 100.0)

    def test_sim_direction_increasing_angle_moves_toward_front(self) -> None:
        x0, _ = unit_from_angle_deg(0)
        x90, _ = unit_from_angle_deg(90)
        x180, _ = unit_from_angle_deg(180)
        self.assertLess(x0, x90)
        self.assertLess(x90, x180)

    def test_knee_winding_is_opposite_hip_winding(self) -> None:
        _, y_hip = unit_from_angle_deg(60)
        _, y_knee = unit_from_knee_angle_deg(60)
        self.assertEqual(round(y_hip, 6), round(-y_knee, 6))

    def test_sim_keeps_knee_joint_angle_when_only_hip_changes(self) -> None:
        dv = sim_store.default_dyn_vars(location_keys=LOCATION_KEYS)

        def interior_knee_deg(front_caps):
            thigh = next(c for c in front_caps if c.name == "front_thigh")
            shin = next(c for c in front_caps if c.name == "front_shin")
            # Interior angle at knee between (knee->hip) and (knee->foot)
            ux = thigh.a[0] - thigh.b[0]
            uy = thigh.a[1] - thigh.b[1]
            vx = shin.b[0] - shin.a[0]
            vy = shin.b[1] - shin.a[1]
            un = math.hypot(ux, uy)
            vn = math.hypot(vx, vy)
            dot = (ux * vx + uy * vy) / (un * vn)
            dot = max(-1.0, min(1.0, dot))
            return math.degrees(math.acos(dot))

        angles1 = {"front_hip": 120.0, "front_knee": 135.0, "rear_hip": 135.0, "rear_knee": 135.0}
        angles2 = {"front_hip": 170.0, "front_knee": 135.0, "rear_hip": 135.0, "rear_knee": 135.0}

        front1, _ = build_leg_capsules_for_side(dv, "left", angles1)
        front2, _ = build_leg_capsules_for_side(dv, "left", angles2)

        self.assertAlmostEqual(interior_knee_deg(front1), interior_knee_deg(front2), places=6)

    def test_sim_calibration_profile_maps_predicted_angle_only(self) -> None:
        module = self.config_app
        state = {"front_left_hip": 200}
        dv = sim_store.default_dyn_vars(location_keys=LOCATION_KEYS)
        dv["calibration_profiles"]["half"] = {
            "fit_mode": "linear_best_fit",
            "measurement_mode": "servo_physical_deg",
            "pairs": [
                {"commanded_deg": 0.0, "actual_deg": 0.0},
                {"commanded_deg": 200.0, "actual_deg": 100.0},
            ],
        }
        dv["joint_calibration_map"]["front_left_hip"] = "half"
        servo_limits = {
            loc_key: servo_limits_from_config_item(module._draft_cfg["locations"][loc_key])
            for loc_key in LOCATION_KEYS
        }
        pack = angles_pack_for_side_from_state("left", state, dv=dv, servo_limits_by_location=servo_limits)
        limits = servo_limits_from_config_item(module._draft_cfg["locations"]["front_left_hip"])
        _logical, physical = self.pca_mod.resolve_logical_and_physical_angle(200, limits)
        self.assertEqual(pack["front_hip"], float(physical) * 0.5)

    def test_default_profiles_include_hip_and_knee_standards(self) -> None:
        dv = sim_store.default_dyn_vars(location_keys=LOCATION_KEYS)
        self.assertIn("hip_standard", dv["calibration_profiles"])
        self.assertIn("knee_standard", dv["calibration_profiles"])
        self.assertEqual(dv["calibration_profiles"]["hip_standard"]["measurement_mode"], "hip_line_relative_deg")
        self.assertEqual(dv["calibration_profiles"]["knee_standard"]["measurement_mode"], "knee_relative_deg")

    def test_hip_and_knee_relative_modes_affect_sim_pack(self) -> None:
        module = self.config_app
        dv = sim_store.default_dyn_vars(location_keys=LOCATION_KEYS)
        dv["calibration_profiles"] = {
            "identity": {
                "fit_mode": "linear_best_fit",
                "measurement_mode": "servo_physical_deg",
                "pairs": [{"commanded_deg": 0, "actual_deg": 0}, {"commanded_deg": 270, "actual_deg": 270}],
            },
            "hip_rel": {
                "fit_mode": "linear_best_fit",
                "measurement_mode": "hip_line_relative_deg",
                "pairs": [{"commanded_deg": 120, "actual_deg": 90}, {"commanded_deg": 140, "actual_deg": 110}],
            },
            "knee_rel": {
                "fit_mode": "linear_best_fit",
                "measurement_mode": "knee_relative_deg",
                "pairs": [{"commanded_deg": 120, "actual_deg": 90}, {"commanded_deg": 160, "actual_deg": 130}],
            },
        }
        dv["joint_calibration_map"] = {loc_key: "identity" for loc_key in LOCATION_KEYS}
        dv["joint_calibration_map"]["front_left_hip"] = "hip_rel"
        dv["joint_calibration_map"]["front_left_knee"] = "knee_rel"

        state = {"front_left_hip": 120, "front_left_knee": 120}
        servo_limits = {
            loc_key: servo_limits_from_config_item(module._draft_cfg["locations"][loc_key])
            for loc_key in LOCATION_KEYS
        }
        pack = angles_pack_for_side_from_state("left", state, dv=dv, servo_limits_by_location=servo_limits)
        # Modes are relative-space; resulting pack values should differ from raw physical passthrough.
        self.assertNotEqual(pack["front_hip"], 120.0)
        self.assertNotEqual(pack["front_knee"], 120.0)

    def test_knee_relative_offset_sweep_avoids_branch_flip(self) -> None:
        module = self.config_app
        dv = sim_store.default_dyn_vars(location_keys=LOCATION_KEYS)
        state = {loc_key: 135 for loc_key in LOCATION_KEYS}
        servo_limits = {
            loc_key: servo_limits_from_config_item(module._draft_cfg["locations"][loc_key])
            for loc_key in LOCATION_KEYS
        }

        def front_shin_world_deg(off: float) -> float:
            local = json.loads(json.dumps(dv))
            local["left"]["front_knee_offset_deg"] = float(off)
            pack = angles_pack_for_side_from_state("left", state, dv=local, servo_limits_by_location=servo_limits)
            a_fh = (pack["front_hip"] + local["left"]["front_hip_offset_deg"]) % 360.0
            a_fk = (pack["front_knee"] + local["left"]["front_knee_offset_deg"]) % 360.0
            vf = rotate_ccw_deg(unit_from_knee_angle_deg(a_fk), a_fh)
            return (math.degrees(math.atan2(vf[1], vf[0])) + 360.0) % 360.0

        a = front_shin_world_deg(120.0)
        b = front_shin_world_deg(130.0)
        # Should not jump by a near-180 branch flip between adjacent offsets.
        delta = abs(((b - a + 180.0) % 360.0) - 180.0)
        self.assertLess(delta, 60.0)

    def test_dynamic_limits_accepts_calibration_schema(self) -> None:
        client = self.config_app.app.test_client()
        cur = client.get("/api/dynamic_limits").get_json()["dynamic_limits"]
        try:
            update = json.loads(json.dumps(cur))
            update["calibration_profiles"] = {
                "identity": {
                    "fit_mode": "linear_best_fit",
                    "measurement_mode": "servo_physical_deg",
                    "pairs": [{"commanded_deg": 0, "actual_deg": 0}],
                },
                "hips": {
                    "fit_mode": "linear_best_fit",
                    "measurement_mode": "hip_line_relative_deg",
                    "pairs": [
                        {"commanded_deg": 0, "actual_deg": 0},
                        {"commanded_deg": 180, "actual_deg": 170},
                    ],
                },
            }
            update["joint_calibration_map"] = {
                "front_left_hip": "hips",
                "front_right_hip": "hips",
                "front_left_knee": "identity",
                "front_right_knee": "identity",
                "rear_left_hip": "hips",
                "rear_right_hip": "hips",
                "rear_left_knee": "identity",
                "rear_right_knee": "identity",
            }
            resp = client.post("/api/dynamic_limits", json={"dynamic_limits": update})
            self.assertEqual(resp.status_code, 200)
            body = resp.get_json()["dynamic_limits"]
            self.assertIn("calibration_profiles", body)
            self.assertIn("joint_calibration_map", body)
            self.assertEqual(body["joint_calibration_map"]["front_left_hip"], "hips")
        finally:
            restore = client.post("/api/dynamic_limits", json={"dynamic_limits": cur})
            self.assertEqual(restore.status_code, 200)

    def test_calibration_does_not_change_hardware_command_output(self) -> None:
        client = self.config_app.app.test_client()
        module = self.config_app

        class _StubPCA:
            def __init__(self) -> None:
                self.calls = []

            def set_channel_angle_deg(self, channel, angle_deg, limits=None):
                self.calls.append((channel, angle_deg, limits))

        original_pca = module.pca
        original_dyn = json.loads(json.dumps(module._dyn_vars))
        stub = _StubPCA()
        module.pca = stub
        try:
            dv = json.loads(json.dumps(module._dyn_vars))
            dv["calibration_profiles"] = {
                "identity": {
                    "fit_mode": "linear_best_fit",
                    "measurement_mode": "servo_physical_deg",
                    "pairs": [{"commanded_deg": 0, "actual_deg": 0}],
                },
                "aggressive": {
                    "fit_mode": "linear_best_fit",
                    "measurement_mode": "servo_physical_deg",
                    "pairs": [
                        {"commanded_deg": 0, "actual_deg": 0},
                        {"commanded_deg": 180, "actual_deg": 90},
                    ],
                },
            }
            dv["joint_calibration_map"] = {k: "identity" for k in module._dyn_vars.get("joint_calibration_map", {}).keys()}
            if not dv["joint_calibration_map"]:
                dv["joint_calibration_map"] = {loc.key: "identity" for loc in module.LOCATIONS}
            dv["joint_calibration_map"]["front_right_hip"] = "aggressive"
            module._dyn_vars = dv

            cmd = client.post(
                "/api/command",
                json={"location": "front_right_hip", "angle_deg": 180, "mode": "normal"},
            )
            self.assertEqual(cmd.status_code, 200)
            data = cmd.get_json()
            self.assertEqual(len(stub.calls), 1)
            _ch, sent_angle, _limits = stub.calls[0]
            self.assertEqual(sent_angle, data["applied_angle"])
        finally:
            module._dyn_vars = original_dyn
            module.pca = original_pca


@unittest.skipUnless(HAVE_FLASK, "Flask not installed in this environment")
class HubApiContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if "smbus2" not in sys.modules:
            fake = types.ModuleType("smbus2")

            class _FakeSMBus:  # pragma: no cover - test bootstrap shim
                def __init__(self, *args, **kwargs):
                    raise OSError("smbus unavailable in test environment")

            fake.SMBus = _FakeSMBus
            sys.modules["smbus2"] = fake

        cls.hub_app = importlib.import_module("hub_app")

    def test_load_program_requires_program_field(self) -> None:
        client = self.hub_app.app.test_client()
        resp = client.post("/api/load_program", json={"num_frames": 200})
        self.assertEqual(resp.status_code, 400)

    def test_load_program_json_and_compile_preview_contract(self) -> None:
        client = self.hub_app.app.test_client()

        cfg = client.post("/api/load_config", json={})
        self.assertEqual(cfg.status_code, 200)

        payload = {
            "program": {
                "program_id": "hub_test",
                "tick_ms": 20,
                "steps": [
                    {
                        "step_id": "lift",
                        "commands": [
                            {"location": "rear_left_hip", "target_angle": 150, "duration_ms": 40},
                            {"location": "rear_right_hip", "target_angle": 150, "duration_ms": 40},
                        ],
                    }
                ],
            }
        }
        load = client.post("/api/load_program_json", json=payload)
        self.assertEqual(load.status_code, 200)
        self.assertTrue(load.get_json()["ok"])

        comp = client.post("/api/compile_program", json={"sparse_targets": True})
        self.assertEqual(comp.status_code, 200)
        self.assertTrue(comp.get_json()["ok"])

        prev = client.get("/api/program_preview?count=5")
        self.assertEqual(prev.status_code, 200)
        body = prev.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("ticks", body)

        telem = client.get("/api/telemetry?tail=5")
        self.assertEqual(telem.status_code, 200)
        self.assertTrue(telem.get_json()["ok"])

    def test_hub_settings_and_heartbeat_contract(self) -> None:
        client = self.hub_app.app.test_client()

        resp = client.post(
            "/api/settings",
            json={"heartbeat_timeout_ms": 2500, "stop_on_clamp": True},
        )
        self.assertEqual(resp.status_code, 200)
        st = resp.get_json()["status"]
        self.assertEqual(float(st["heartbeat_timeout_ms"]), 2500.0)
        self.assertTrue(bool(st["stop_on_clamp"]))

        hb = client.post("/api/heartbeat", json={})
        self.assertEqual(hb.status_code, 200)
        self.assertTrue(hb.get_json()["ok"])

    def test_settings_validation_rejects_invalid_values(self) -> None:
        client = self.hub_app.app.test_client()

        resp = client.post("/api/settings", json={"heartbeat_timeout_ms": -1})
        self.assertEqual(resp.status_code, 400)

        resp = client.post("/api/settings", json={"heartbeat_timeout_ms": 120000})
        self.assertEqual(resp.status_code, 400)

        resp = client.post("/api/settings", json={"heartbeat_timeout_ms": 0, "stop_on_clamp": False})
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
