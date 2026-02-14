from __future__ import annotations

import json
import importlib
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
