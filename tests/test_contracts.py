from __future__ import annotations

import json
import importlib
import math
import sys
import types
import unittest
from pathlib import Path

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
        pack = module._angles_pack_for_side_from_state("right", state)
        self.assertEqual(pack["front_hip"], 100.0)
        self.assertEqual(pack["front_knee"], 90.0)
        self.assertEqual(pack["rear_hip"], 120.0)
        self.assertEqual(pack["rear_knee"], 100.0)

    def test_sim_direction_increasing_angle_moves_toward_front(self) -> None:
        module = self.config_app
        x0, _ = module._unit_from_angle_deg(0)
        x90, _ = module._unit_from_angle_deg(90)
        x180, _ = module._unit_from_angle_deg(180)
        self.assertLess(x0, x90)
        self.assertLess(x90, x180)

    def test_knee_winding_is_opposite_hip_winding(self) -> None:
        module = self.config_app
        _, y_hip = module._unit_from_angle_deg(60)
        _, y_knee = module._unit_from_knee_angle_deg(60)
        self.assertEqual(round(y_hip, 6), round(-y_knee, 6))

    def test_sim_keeps_knee_joint_angle_when_only_hip_changes(self) -> None:
        module = self.config_app
        dv = module._default_dyn_vars()

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

        front1, _ = module._build_leg_capsules_for_side(dv, "left", angles1)
        front2, _ = module._build_leg_capsules_for_side(dv, "left", angles2)

        self.assertAlmostEqual(interior_knee_deg(front1), interior_knee_deg(front2), places=6)


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

    def test_settings_validation_rejects_invalid_values(self) -> None:
        client = self.hub_app.app.test_client()

        resp = client.post("/api/settings", json={"slice_period_ms": 0})
        self.assertEqual(resp.status_code, 400)

        resp = client.post("/api/settings", json={"overrun_warn_ms": -1})
        self.assertEqual(resp.status_code, 400)

        resp = client.post("/api/settings", json={"slice_period_ms": 20, "overrun_warn_ms": 0.5})
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
