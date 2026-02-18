from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

from flask import Flask, jsonify, request, send_from_directory

from hardware.pca9685 import PCA9685, ServoLimits, resolve_logical_and_physical_angle
from sim_core import sim_store
from sim_core.collision import collision_snapshot_for_state as sim_collision_snapshot_for_state
from sim_core.engine import SimulationEngine


# -----------------------------
# Doggo constants (config app)
# -----------------------------
ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 270

CHANNEL_MIN = 0
CHANNEL_MAX = 15

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "config_file.json"
LOCATION_ORDER_FILE = CONFIG_DIR / "location_order.json"
CHANNEL_NOTES_FILE = CONFIG_DIR / "channel_notes.json"
CHANNEL_NOTES_MAX_CHARS = 50_000

DYN_VARS_FILE = CONFIG_DIR / "dynamic_limits.json"
STANCES_DIR = BASE_DIR / "stances"
DEFAULT_STANCE_NAME = "default"
STANCE_STEP_DELAY_MS = 250

# PCA9685 constants (locked frequency is inside driver)
I2C_BUS = 1
I2C_ADDR = 0x40


# -----------------------------
# Fixed locations (backend-owned)
# -----------------------------
@dataclass(frozen=True)
class Location:
    key: str
    label: str


LOCATIONS: List[Location] = [
    Location("front_left_hip", "Front Left Hip"),
    Location("front_left_knee", "Front Left Knee"),
    Location("front_right_hip", "Front Right Hip"),
    Location("front_right_knee", "Front Right Knee"),
    Location("rear_left_hip", "Rear Left Hip"),
    Location("rear_left_knee", "Rear Left Knee"),
    Location("rear_right_hip", "Rear Right Hip"),
    Location("rear_right_knee", "Rear Right Knee"),
]


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _all_channels() -> List[int]:
    return list(range(CHANNEL_MIN, CHANNEL_MAX + 1))


def _default_limits() -> Dict[str, Any]:
    return {"deg_min": 0, "deg_max": 270, "invert": False}


def _empty_config() -> Dict[str, Any]:
    """
    Canonical config structure (version 2):
    {
      "version": 2,
      "locations": {
        "<loc_key>": {
          "channel": <int or null>,
          "limits": { "deg_min": int, "deg_max": int, "invert": bool }
        },
        ...
      }
    }
    """
    return {
        "version": 2,
        "locations": {
            loc.key: {"channel": None, "limits": _default_limits()}
            for loc in LOCATIONS
        },
    }


def _ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_or_init_location_order() -> List[str]:
    _ensure_config_dir()

    default_order = [loc.key for loc in LOCATIONS]

    if not LOCATION_ORDER_FILE.exists():
        LOCATION_ORDER_FILE.write_text(json.dumps({"version": 1, "position_order": default_order}, indent=2))
        return default_order

    try:
        data = json.loads(LOCATION_ORDER_FILE.read_text())
        order = data.get("position_order", [])
        if not isinstance(order, list):
            return default_order
        order = [str(x) for x in order]

        known = {loc.key for loc in LOCATIONS}
        cleaned = [k for k in order if k in known]
        for k in default_order:
            if k not in cleaned:
                cleaned.append(k)

        if cleaned != order:
            LOCATION_ORDER_FILE.write_text(json.dumps({"version": 1, "position_order": cleaned}, indent=2))
        return cleaned
    except Exception:
        LOCATION_ORDER_FILE.write_text(json.dumps({"version": 1, "position_order": default_order}, indent=2))
        return default_order


def _load_saved_config() -> Dict[str, Any]:
    _ensure_config_dir()

    if not CONFIG_FILE.exists():
        return _empty_config()

    try:
        data = json.loads(CONFIG_FILE.read_text())
        if int(data.get("version", 0)) != 2:
            return _empty_config()

        out = _empty_config()
        locs = data.get("locations", {})
        if not isinstance(locs, dict):
            return out

        for loc in LOCATIONS:
            item = locs.get(loc.key, {})
            if not isinstance(item, dict):
                continue

            # channel
            ch = item.get("channel", None)
            if ch is None:
                out["locations"][loc.key]["channel"] = None
            else:
                try:
                    ch_i = int(ch)
                    if CHANNEL_MIN <= ch_i <= CHANNEL_MAX:
                        out["locations"][loc.key]["channel"] = ch_i
                    else:
                        out["locations"][loc.key]["channel"] = None
                except Exception:
                    out["locations"][loc.key]["channel"] = None

            # limits
            lim = item.get("limits", {})
            if not isinstance(lim, dict):
                lim = {}

            try:
                raw_min = int(lim.get("deg_min", 0))
            except Exception:
                raw_min = 0
            try:
                raw_max = int(lim.get("deg_max", 270))
            except Exception:
                raw_max = 270

            deg_min = _clamp_int(raw_min, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
            deg_max = _clamp_int(raw_max, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
            if deg_max < deg_min:
                deg_min, deg_max = deg_max, deg_min
            invert = bool(lim.get("invert", False))

            out["locations"][loc.key]["limits"] = {"deg_min": deg_min, "deg_max": deg_max, "invert": invert}

        return out
    except Exception:
        return _empty_config()


def _save_config(cfg: Dict[str, Any]) -> None:
    _ensure_config_dir()
    cfg = dict(cfg)
    cfg["version"] = 2
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def _compute_available_channels(cfg: Dict[str, Any]) -> List[int]:
    used = set()
    for loc in LOCATIONS:
        ch = cfg["locations"][loc.key]["channel"]
        if ch is not None:
            used.add(int(ch))
    return [ch for ch in _all_channels() if ch not in used]


def _dirty(draft: Dict[str, Any], saved: Dict[str, Any]) -> bool:
    return draft != saved


def _validate_no_duplicate_channels(cfg: Dict[str, Any]) -> Tuple[bool, str]:
    seen: Dict[int, str] = {}
    for loc in LOCATIONS:
        ch = cfg["locations"][loc.key]["channel"]
        if ch is None:
            continue
        ch_i = int(ch)
        if ch_i in seen:
            return False, f"Duplicate channel {ch_i} assigned to '{seen[ch_i]}' and '{loc.key}'."
        seen[ch_i] = loc.key
    return True, ""


def _get_limits(cfg: Dict[str, Any], loc_key: str) -> ServoLimits:
    lim = cfg["locations"][loc_key]["limits"]
    return ServoLimits(
        deg_min=int(lim["deg_min"]),
        deg_max=int(lim["deg_max"]),
        invert=bool(lim["invert"]),
    )


# -----------------------------
# Channel notes helpers
# -----------------------------
def _load_channel_notes() -> str:
    _ensure_config_dir()
    if not CHANNEL_NOTES_FILE.exists():
        return ""
    try:
        data = json.loads(CHANNEL_NOTES_FILE.read_text())
        notes = data.get("notes", "")
        if not isinstance(notes, str):
            return ""
        return notes
    except Exception:
        return ""


def _save_channel_notes(notes: str) -> None:
    _ensure_config_dir()
    payload = {"version": 1, "notes": notes}
    CHANNEL_NOTES_FILE.write_text(json.dumps(payload, indent=2))


# -----------------------------
# Dynamic limit variables (capsule model)
# -----------------------------
def _stance_location_keys() -> List[str]:
    return [loc.key for loc in LOCATIONS]


def _servo_limits_by_location(cfg: Dict[str, Any]) -> Dict[str, ServoLimits]:
    return {loc.key: _get_limits(cfg, loc.key) for loc in LOCATIONS}


# -----------------------------
# App init
# -----------------------------
app = Flask(__name__, static_folder="static")

_location_order: List[str] = _load_or_init_location_order()

_saved_cfg: Dict[str, Any] = _load_saved_config()
_draft_cfg: Dict[str, Any] = json.loads(json.dumps(_saved_cfg))

_dyn_vars: Dict[str, Any] = sim_store.load_dyn_vars(DYN_VARS_FILE, location_keys=_stance_location_keys())

# Hardware driver (optional)
pca: Optional[PCA9685] = None
_pca_error: Optional[str] = None
try:
    pca = PCA9685(i2c_bus=I2C_BUS, address=I2C_ADDR)
except Exception as e:
    pca = None
    _pca_error = str(e)

# Angle state
DEFAULT_NEUTRAL = 135
_hw_angles: Dict[str, int] = {loc.key: DEFAULT_NEUTRAL for loc in LOCATIONS}
_sim_angles: Dict[str, int] = {loc.key: DEFAULT_NEUTRAL for loc in LOCATIONS}
_sim_engine = SimulationEngine(dynamic_limits=_dyn_vars, servo_limits_by_location=_servo_limits_by_location(_draft_cfg))
_sim_engine.set_state("normal", _hw_angles)
_sim_engine.set_state("test", _sim_angles)


# -----------------------------
# Static UI
# -----------------------------
@app.get("/")
def index():
    return send_from_directory("static", "config_page.html")


def _sync_sim_engine_inputs(include_states: bool = True) -> None:
    _sim_engine.set_dynamic_limits(_dyn_vars)
    _sim_engine.set_servo_limits_by_location(_servo_limits_by_location(_draft_cfg))
    if include_states:
        _sim_engine.set_state("normal", _hw_angles)
        _sim_engine.set_state("test", _sim_angles)


def _execute_servo_command(loc_key: str, requested_angle: int, mode: str) -> Tuple[bool, Dict[str, Any], int]:
    global _hw_angles, _sim_angles

    if loc_key not in _draft_cfg["locations"]:
        return False, {"ok": False, "error": f"Unknown location '{loc_key}'"}, 400

    if mode not in ("normal", "test"):
        mode = "normal"

    requested = _clamp_int(int(requested_angle), ANGLE_MIN_DEG, ANGLE_MAX_DEG)

    limits = _get_limits(_draft_cfg, loc_key)

    _sync_sim_engine_inputs(include_states=True)
    state_name = "test" if mode == "test" else "normal"
    try:
        sim_out = _sim_engine.apply_command(state_name=state_name, loc_key=loc_key, requested_angle=requested)
    except KeyError:
        return False, {"ok": False, "error": f"Unknown location '{loc_key}'"}, 400

    # In normal mode, require channel + hardware
    if mode == "normal":
        ch = _draft_cfg["locations"][loc_key]["channel"]
        if ch is None:
            _sim_engine.set_state("normal", _hw_angles)
            return False, {"ok": False, "error": f"Location '{loc_key}' is unassigned"}, 409
        if pca is None:
            _sim_engine.set_state("normal", _hw_angles)
            return False, {"ok": False, "error": "Hardware not available (PCA9685 init failed)."}, 503

        try:
            # Keep high-level logic inversion-agnostic; hardware output handles invert.
            pca.set_channel_angle_deg(int(ch), int(sim_out.applied_angle), limits=limits)
        except Exception as e:
            _sim_engine.set_state("normal", _hw_angles)
            return False, {"ok": False, "error": f"Hardware command failed: {e}"}, 503
        _hw_angles = _sim_engine.get_state("normal")
    else:
        _sim_angles = _sim_engine.get_state("test")

    return (
        True,
        {
            "ok": True,
            "mode": mode,
            "requested_angle": int(sim_out.requested_angle),
            "travel_applied_angle": int(sim_out.travel_applied_angle),
            "applied_angle": int(sim_out.applied_angle),
            "clamped": bool(sim_out.clamped),
            "clamp_reason": sim_out.clamp_reason,
            "collision": sim_out.collision,
        },
        200,
    )


def _collision_snapshot_for_state(state_angles: Dict[str, int]) -> Dict[str, Any]:
    return sim_collision_snapshot_for_state(
        dv=_dyn_vars,
        state_angles=state_angles,
        servo_limits_by_location=_servo_limits_by_location(_draft_cfg),
    )


# -----------------------------
# API
# -----------------------------
@app.get("/api/config")
def api_get_config():
    available = _compute_available_channels(_draft_cfg)
    collision_status = {
        "normal": _collision_snapshot_for_state(_hw_angles),
        "test": _collision_snapshot_for_state(_sim_angles),
    }
    return jsonify(
        {
            "locations": [{"key": loc.key, "label": loc.label} for loc in LOCATIONS],
            "location_order": _location_order,
            "draft": _draft_cfg,
            "available_channels": available,
            "save_enabled": _dirty(_draft_cfg, _saved_cfg),
            "reset_enabled": _dirty(_draft_cfg, _saved_cfg),
            "angle_range": {"min": ANGLE_MIN_DEG, "max": ANGLE_MAX_DEG},
            "channel_range": {"min": CHANNEL_MIN, "max": CHANNEL_MAX},
            "hardware_available": pca is not None,
            "hardware_error": _pca_error,
            "dynamic_limits": _dyn_vars,
            "hw_angles": _hw_angles,
            "sim_angles": _sim_angles,
            "collision_status": collision_status,
        }
    )


@app.get("/api/dynamic_limits")
def api_get_dynamic_limits():
    return jsonify({"ok": True, "dynamic_limits": _dyn_vars})


@app.post("/api/dynamic_limits")
def api_set_dynamic_limits():
    global _dyn_vars

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    dv = data.get("dynamic_limits", None)
    if not isinstance(dv, dict):
        return jsonify({"ok": False, "error": "dynamic_limits must be an object"}), 400

    # Start from current to avoid accidental missing keys wiping config
    merged = json.loads(json.dumps(_dyn_vars))

    # active_side
    active_side = str(dv.get("active_side", merged.get("active_side", "left"))).strip().lower()
    if active_side not in ("left", "right"):
        active_side = merged.get("active_side", "left")
    merged["active_side"] = active_side

    # globals
    if "search_step_deg" in dv:
        merged["search_step_deg"] = _clamp_int(sim_store.coerce_int(dv.get("search_step_deg"), merged["search_step_deg"]), 1, 45)
    if "search_max_iters" in dv:
        merged["search_max_iters"] = _clamp_int(sim_store.coerce_int(dv.get("search_max_iters"), merged["search_max_iters"]), 1, 5000)

    # sides
    side_template = sim_store.default_dyn_side()
    for side in ("left", "right"):
        if side not in dv:
            continue
        side_in = dv.get(side)
        if not isinstance(side_in, dict):
            continue
        side_out = merged.get(side, side_template.copy())
        for k in side_template.keys():
            if k in side_in:
                side_out[k] = sim_store.coerce_float(side_in.get(k), side_out.get(k, 0.0))
        merged[side] = side_out

    # calibration profiles
    if "calibration_profiles" in dv:
        profs_in = dv.get("calibration_profiles")
        if not isinstance(profs_in, dict):
            return jsonify({"ok": False, "error": "calibration_profiles must be an object"}), 400
        next_profiles: Dict[str, Dict[str, Any]] = {}
        for name_raw, prof_raw in profs_in.items():
            name = str(name_raw).strip()
            if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name) is None:
                return jsonify({"ok": False, "error": f"invalid profile name '{name}'"}), 400
            if not isinstance(prof_raw, dict):
                return jsonify({"ok": False, "error": f"profile '{name}' must be an object"}), 400
            mode = str(prof_raw.get("measurement_mode", "servo_physical_deg")).strip().lower()
            if mode not in ("servo_physical_deg", "hip_line_relative_deg", "knee_relative_deg"):
                return jsonify({"ok": False, "error": f"profile '{name}'.measurement_mode is invalid"}), 400
            pairs_raw = prof_raw.get("pairs", [])
            if not isinstance(pairs_raw, list):
                return jsonify({"ok": False, "error": f"profile '{name}'.pairs must be an array"}), 400
            pairs_out: List[Dict[str, float]] = []
            for item in pairs_raw:
                if not isinstance(item, dict):
                    return jsonify({"ok": False, "error": f"profile '{name}'.pairs entries must be objects"}), 400
                c = sim_store.coerce_float(item.get("commanded_deg"), 0.0)
                a = sim_store.coerce_float(item.get("actual_deg"), 0.0)
                pairs_out.append(
                    {
                        "commanded_deg": float(max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, c))),
                        "actual_deg": float(max(ANGLE_MIN_DEG, min(ANGLE_MAX_DEG, a))),
                    }
                )
            next_profiles[name] = {"fit_mode": "linear_best_fit", "measurement_mode": mode, "pairs": pairs_out}
        if "identity" not in next_profiles:
            next_profiles["identity"] = sim_store.default_calibration_profile()
        merged["calibration_profiles"] = next_profiles

    # per-joint profile assignment
    if "joint_calibration_map" in dv:
        map_in = dv.get("joint_calibration_map")
        if not isinstance(map_in, dict):
            return jsonify({"ok": False, "error": "joint_calibration_map must be an object"}), 400
        map_out = dict(
            merged.get("joint_calibration_map", sim_store.default_joint_calibration_map(location_keys=_stance_location_keys()))
        )
        profiles = merged.get("calibration_profiles", {})
        if not isinstance(profiles, dict):
            profiles = {"identity": sim_store.default_calibration_profile()}
        for loc in LOCATIONS:
            if loc.key not in map_in:
                continue
            prof_name = str(map_in.get(loc.key, "identity")).strip()
            if prof_name not in profiles:
                return jsonify({"ok": False, "error": f"unknown profile '{prof_name}' for '{loc.key}'"}), 400
            map_out[loc.key] = prof_name
        merged["joint_calibration_map"] = map_out

    # guarantee full joint map and valid references
    profiles = merged.get("calibration_profiles", {})
    if not isinstance(profiles, dict):
        profiles = {"identity": sim_store.default_calibration_profile()}
    if "identity" not in profiles:
        profiles["identity"] = sim_store.default_calibration_profile()
    if "hip_standard" not in profiles:
        profiles["hip_standard"] = sim_store.default_hip_standard_profile()
    if "knee_standard" not in profiles:
        profiles["knee_standard"] = sim_store.default_knee_standard_profile()
    merged["calibration_profiles"] = profiles

    jmap = merged.get("joint_calibration_map", {})
    if not isinstance(jmap, dict):
        jmap = {}
    complete_map: Dict[str, str] = {}
    for loc in LOCATIONS:
        p = str(jmap.get(loc.key, "identity")).strip()
        complete_map[loc.key] = p if p in profiles else "identity"
    merged["joint_calibration_map"] = complete_map

    for side in ("left", "right"):
        err = sim_store.validate_dyn_side_values(side, merged[side])
        if err is not None:
            return jsonify({"ok": False, "error": err}), 400

    merged["version"] = 2
    sim_store.save_dyn_vars(DYN_VARS_FILE, merged)
    _dyn_vars = merged
    _sync_sim_engine_inputs(include_states=False)
    return jsonify({"ok": True, "dynamic_limits": _dyn_vars})


@app.get("/api/stances")
def api_get_stances():
    STANCES_DIR.mkdir(parents=True, exist_ok=True)
    stances: List[Dict[str, str]] = []
    invalid: List[Dict[str, str]] = []

    for path in sorted(STANCES_DIR.glob("*.json")):
        name = path.stem
        ok, _, err = sim_store.load_stance(STANCES_DIR, name, location_keys=_stance_location_keys())
        if ok:
            stances.append({"name": name, "file": path.name})
        else:
            invalid.append({"name": name, "error": err})

    return jsonify({"ok": True, "stances": stances, "invalid": invalid, "default_stance": DEFAULT_STANCE_NAME})


@app.post("/api/stance/activate")
def api_activate_stance():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    stance_name = str(data.get("stance", DEFAULT_STANCE_NAME)).strip()
    mode = str(data.get("mode", "normal")).strip().lower()
    if mode not in ("normal", "test"):
        mode = "normal"

    ok, angles, err = sim_store.load_stance(STANCES_DIR, stance_name, location_keys=_stance_location_keys())
    if not ok:
        return jsonify({"ok": False, "error": err}), 400

    sequence = [k for k in _location_order if k in angles]
    for k in _stance_location_keys():
        if k not in sequence:
            sequence.append(k)

    results: List[Dict[str, Any]] = []
    delay_s = max(0.0, float(STANCE_STEP_DELAY_MS) / 1000.0)

    for idx, loc_key in enumerate(sequence):
        cmd_ok, payload, status = _execute_servo_command(loc_key, int(angles[loc_key]), mode)
        results.append({"location": loc_key, **payload})
        if not cmd_ok:
            return (
                jsonify(
                    {
                        "ok": False,
                        "stance": stance_name,
                        "mode": mode,
                        "delay_ms": STANCE_STEP_DELAY_MS,
                        "completed_count": len(results),
                        "results": results,
                        "hw_angles": _hw_angles,
                        "sim_angles": _sim_angles,
                        "error": payload.get("error", "stance activation failed"),
                    }
                ),
                status,
            )

        if idx < len(sequence) - 1 and delay_s > 0:
            time.sleep(delay_s)

    return jsonify(
        {
            "ok": True,
            "stance": stance_name,
            "mode": mode,
            "delay_ms": STANCE_STEP_DELAY_MS,
            "completed_count": len(results),
            "results": results,
            "hw_angles": _hw_angles,
            "sim_angles": _sim_angles,
        }
    )


@app.get("/api/channel_notes")
def api_get_channel_notes():
    return jsonify({"ok": True, "notes": _load_channel_notes()})


@app.post("/api/channel_notes")
def api_set_channel_notes():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    notes = data.get("notes", "")
    if not isinstance(notes, str):
        return jsonify({"ok": False, "error": "notes must be a string"}), 400
    if len(notes) > CHANNEL_NOTES_MAX_CHARS:
        return jsonify({"ok": False, "error": f"notes too large (max {CHANNEL_NOTES_MAX_CHARS} chars)"}), 413
    _save_channel_notes(notes)
    return jsonify({"ok": True})


@app.post("/api/channel")
def api_set_channel():
    global _draft_cfg

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    loc_key = str(data.get("location", "")).strip()
    if loc_key not in _draft_cfg["locations"]:
        return jsonify({"ok": False, "error": f"Unknown location '{loc_key}'"}), 400

    channel = data.get("channel", None)
    if channel is not None:
        try:
            channel = int(channel)
        except Exception:
            return jsonify({"ok": False, "error": "channel must be an integer or null"}), 400
        if not (CHANNEL_MIN <= channel <= CHANNEL_MAX):
            return jsonify({"ok": False, "error": f"channel out of range {CHANNEL_MIN}..{CHANNEL_MAX}"}), 400

    current = _draft_cfg["locations"][loc_key]["channel"]

    if channel is not None and channel != current:
        available = set(_compute_available_channels(_draft_cfg))
        if channel not in available:
            return jsonify({"ok": False, "error": f"Channel {channel} is already assigned elsewhere"}), 409

    _draft_cfg["locations"][loc_key]["channel"] = channel

    return jsonify(
        {
            "ok": True,
            "draft": _draft_cfg,
            "available_channels": _compute_available_channels(_draft_cfg),
            "save_enabled": _dirty(_draft_cfg, _saved_cfg),
            "reset_enabled": _dirty(_draft_cfg, _saved_cfg),
        }
    )


@app.post("/api/limits")
def api_set_limits():
    global _draft_cfg

    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    loc_key = str(data.get("location", "")).strip()
    if loc_key not in _draft_cfg["locations"]:
        return jsonify({"ok": False, "error": f"Unknown location '{loc_key}'"}), 400

    raw_min = data.get("deg_min")
    raw_max = data.get("deg_max")
    if raw_min is None or raw_max is None:
        return jsonify({"ok": False, "error": "deg_min/deg_max are required"}), 400

    try:
        deg_min = _clamp_int(int(raw_min), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        deg_max = _clamp_int(int(raw_max), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    except Exception:
        return jsonify({"ok": False, "error": "deg_min/deg_max must be integers"}), 400

    if deg_max < deg_min:
        deg_min, deg_max = deg_max, deg_min

    invert = bool(data.get("invert", False))

    _draft_cfg["locations"][loc_key]["limits"] = {"deg_min": deg_min, "deg_max": deg_max, "invert": invert}

    return jsonify(
        {
            "ok": True,
            "draft": _draft_cfg,
            "save_enabled": _dirty(_draft_cfg, _saved_cfg),
            "reset_enabled": _dirty(_draft_cfg, _saved_cfg),
        }
    )


@app.post("/api/reset")
def api_reset():
    global _draft_cfg, _saved_cfg
    _draft_cfg = json.loads(json.dumps(_saved_cfg))
    return jsonify(
        {
            "ok": True,
            "draft": _draft_cfg,
            "available_channels": _compute_available_channels(_draft_cfg),
            "save_enabled": False,
            "reset_enabled": False,
        }
    )


@app.post("/api/save")
def api_save():
    global _draft_cfg, _saved_cfg

    ok, msg = _validate_no_duplicate_channels(_draft_cfg)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 409

    _save_config(_draft_cfg)
    _saved_cfg = json.loads(json.dumps(_draft_cfg))

    return jsonify({"ok": True, "save_enabled": False, "reset_enabled": False})


@app.post("/api/command")
def api_command():
    """
    Request body:
      {
        "location": "<loc_key>",
        "angle_deg": <int>,
        "mode": "normal" | "test"   (optional; default "normal")
      }

    Behavior:
      - Always applies logical travel clamp (ServoLimits window)
      - Always applies collision clamp (capsule model + step search)
      - In test mode: does NOT send PWM, but updates sim state (invert-agnostic)
      - In normal mode: sends PWM (requires PCA); hardware layer applies invert
      - Returns applied_angle; UI should update input field to it
    """
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be an object"}), 400

    loc_key = str(data.get("location", "")).strip()

    raw_angle = data.get("angle_deg")
    if raw_angle is None:
        return jsonify({"ok": False, "error": "angle_deg is required"}), 400

    try:
        requested = int(raw_angle)
    except Exception:
        return jsonify({"ok": False, "error": "angle_deg must be an integer"}), 400

    mode = str(data.get("mode", "normal")).strip().lower()
    ok, payload, status = _execute_servo_command(loc_key, requested, mode)
    return jsonify(payload), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
