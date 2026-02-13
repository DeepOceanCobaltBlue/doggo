from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

from flask import Flask, jsonify, request, send_from_directory

from hardware.pca9685 import PCA9685, ServoLimits


# -----------------------------
# Doggo constants (config app)
# -----------------------------
ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 270

CHANNEL_MIN = 0
CHANNEL_MAX = 15

CONFIG_DIR = Path("config")
CONFIG_FILE = CONFIG_DIR / "config_file.json"
LOCATION_ORDER_FILE = CONFIG_DIR / "location_order.json"
CHANNEL_NOTES_FILE = CONFIG_DIR / "channel_notes.json"
CHANNEL_NOTES_MAX_CHARS = 50_000

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
    Location("back_left_hip", "Back Left Hip"),
    Location("back_left_knee", "Back Left Knee"),
    Location("back_right_hip", "Back Right Hip"),
    Location("back_right_knee", "Back Right Knee"),
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

            deg_min = _clamp_int(int(lim.get("deg_min", 0)), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
            deg_max = _clamp_int(int(lim.get("deg_max", 270)), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
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
# App init
# -----------------------------
app = Flask(__name__, static_folder="static")

_location_order: List[str] = _load_or_init_location_order()

_saved_cfg: Dict[str, Any] = _load_saved_config()
_draft_cfg: Dict[str, Any] = json.loads(json.dumps(_saved_cfg))

# Hardware driver (optional)
pca: Optional[PCA9685] = None
_pca_error: Optional[str] = None
try:
    pca = PCA9685(i2c_bus=I2C_BUS, address=I2C_ADDR)
except Exception as e:
    pca = None
    _pca_error = str(e)


# -----------------------------
# Static UI
# -----------------------------
@app.get("/")
def index():
    return send_from_directory("static", "config_page.html")


# -----------------------------
# API
# -----------------------------
@app.get("/api/config")
def api_get_config():
    global _saved_cfg, _draft_cfg
    available = _compute_available_channels(_draft_cfg)
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
        }
    )


@app.get("/api/channel_notes")
def api_get_channel_notes():
    return jsonify({"ok": True, "notes": _load_channel_notes()})


@app.post("/api/channel_notes")
def api_set_channel_notes():
    data = request.get_json(force=True)
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
    loc_key = str(data.get("location", "")).strip()
    if loc_key not in _draft_cfg["locations"]:
        return jsonify({"ok": False, "error": f"Unknown location '{loc_key}'"}), 400

    try:
        deg_min = _clamp_int(int(data.get("deg_min")), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        deg_max = _clamp_int(int(data.get("deg_max")), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
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
    global _draft_cfg, pca

    if pca is None:
        return jsonify({"ok": False, "error": "Hardware not available (PCA9685 init failed)."}), 503

    data = request.get_json(force=True)
    loc_key = str(data.get("location", "")).strip()
    if loc_key not in _draft_cfg["locations"]:
        return jsonify({"ok": False, "error": f"Unknown location '{loc_key}'"}), 400

    ch = _draft_cfg["locations"][loc_key]["channel"]
    if ch is None:
        return jsonify({"ok": False, "error": f"Location '{loc_key}' is unassigned"}), 409

    try:
        angle = int(data.get("angle_deg"))
    except Exception:
        return jsonify({"ok": False, "error": "angle_deg must be an integer"}), 400

    angle = _clamp_int(angle, ANGLE_MIN_DEG, ANGLE_MAX_DEG)

    limits = _get_limits(_draft_cfg, loc_key)
    pca.set_channel_angle_deg(int(ch), angle, limits=limits)

    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
