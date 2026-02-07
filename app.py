from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

from flask import Flask, jsonify, request, send_from_directory

from hardware.pca9685 import PCA9685  # your module path


# -----------------------------
# Doggo constants
# -----------------------------
ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 270

CHANNEL_MIN = 0
CHANNEL_MAX = 15

CONFIG_PATH = Path("doggo_config.json")

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


def _empty_mapping() -> Dict[str, Optional[int]]:
    return {loc.key: None for loc in LOCATIONS}


def _load_saved_mapping() -> Dict[str, Optional[int]]:
    """
    Loads mapping from disk. If missing/invalid, returns empty mapping.
    File format:
      {
        "version": 1,
        "mapping": { "front_left_hip": 0, ... }
      }
    """
    if not CONFIG_PATH.exists():
        return _empty_mapping()

    try:
        data = json.loads(CONFIG_PATH.read_text())
        mapping = data.get("mapping", {})
        out = _empty_mapping()
        for loc in LOCATIONS:
            val = mapping.get(loc.key, None)
            if val is None:
                out[loc.key] = None
            else:
                ch = int(val)
                if CHANNEL_MIN <= ch <= CHANNEL_MAX:
                    out[loc.key] = ch
                else:
                    out[loc.key] = None
        return out
    except Exception:
        return _empty_mapping()


def _save_mapping(mapping: Dict[str, Optional[int]]) -> None:
    payload = {"version": 1, "mapping": mapping}
    CONFIG_PATH.write_text(json.dumps(payload, indent=2))


def _compute_available(mapping: Dict[str, Optional[int]]) -> List[int]:
    """
    Returns channels not currently assigned in this mapping.
    """
    used = {ch for ch in mapping.values() if ch is not None}
    return [ch for ch in _all_channels() if ch not in used]


def _dirty(draft: Dict[str, Optional[int]], saved: Dict[str, Optional[int]]) -> bool:
    return draft != saved


def _validate_mapping_no_dupes(mapping: Dict[str, Optional[int]]) -> Tuple[bool, str]:
    """
    Even though UI and backend enforce the pool, keep a final guard.
    """
    seen = {}
    for k, ch in mapping.items():
        if ch is None:
            continue
        if ch in seen:
            return False, f"Duplicate channel {ch} assigned to '{seen[ch]}' and '{k}'."
        seen[ch] = k
    return True, ""


# -----------------------------
# App init
# -----------------------------
app = Flask(__name__, static_folder="static")

# saved vs draft state (draft is what dropdowns edit)
_saved_mapping: Dict[str, Optional[int]] = _load_saved_mapping()
_draft_mapping: Dict[str, Optional[int]] = dict(_saved_mapping)

# Hardware driver (single instance)
pca = PCA9685(i2c_bus=I2C_BUS, address=I2C_ADDR)


# -----------------------------
# Static UI
# -----------------------------
@app.get("/")
def index():
    # module 3 will create static/index.html
    return send_from_directory("static", "index.html")


# -----------------------------
# API
# -----------------------------
@app.get("/api/config")
def api_get_config():
    """
    Returns:
      - locations (key + full label)
      - draft mapping
      - availability pool (global)
      - save/reset enabled (derived)
    """
    global _saved_mapping, _draft_mapping
    available = _compute_available(_draft_mapping)
    return jsonify({
        "locations": [{"key": loc.key, "label": loc.label} for loc in LOCATIONS],
        "draft_mapping": _draft_mapping,
        "available_channels": available,
        "save_enabled": _dirty(_draft_mapping, _saved_mapping),
        "reset_enabled": _dirty(_draft_mapping, _saved_mapping),
        "angle_range": {"min": ANGLE_MIN_DEG, "max": ANGLE_MAX_DEG},
        "channel_range": {"min": CHANNEL_MIN, "max": CHANNEL_MAX},
    })


@app.post("/api/mapping")
def api_set_mapping():
    """
    Body: { "location": "<key>", "channel": <0..15 or null> }

    Server enforces single availability pool:
      - You can set location->None (unassign)
      - You can set location->ch only if ch is currently available OR already assigned to this location
    """
    global _draft_mapping

    data = request.get_json(force=True)
    loc_key = str(data.get("location", "")).strip()
    if loc_key not in _draft_mapping:
        return jsonify({"ok": False, "error": f"Unknown location '{loc_key}'"}), 400

    channel = data.get("channel", None)
    if channel is not None:
        try:
            channel = int(channel)
        except Exception:
            return jsonify({"ok": False, "error": "Channel must be an integer or null"}), 400
        if not (CHANNEL_MIN <= channel <= CHANNEL_MAX):
            return jsonify({"ok": False, "error": f"Channel out of range {CHANNEL_MIN}..{CHANNEL_MAX}"}), 400

    current = _draft_mapping[loc_key]

    # If assigning a channel: must be available globally OR already owned by this location
    if channel is not None and channel != current:
        available = set(_compute_available(_draft_mapping))
        if channel not in available:
            return jsonify({"ok": False, "error": f"Channel {channel} is already assigned elsewhere"}), 409

    # Apply update
    _draft_mapping[loc_key] = channel

    # Return updated pool and enabled state
    return jsonify({
        "ok": True,
        "draft_mapping": _draft_mapping,
        "available_channels": _compute_available(_draft_mapping),
        "save_enabled": _dirty(_draft_mapping, _saved_mapping),
        "reset_enabled": _dirty(_draft_mapping, _saved_mapping),
    })


@app.post("/api/reset")
def api_reset():
    """
    Discard unsaved changes: draft <- saved
    """
    global _draft_mapping, _saved_mapping
    _draft_mapping = dict(_saved_mapping)
    return jsonify({
        "ok": True,
        "draft_mapping": _draft_mapping,
        "available_channels": _compute_available(_draft_mapping),
        "save_enabled": False,
        "reset_enabled": False,
    })


@app.post("/api/save")
def api_save():
    """
    Persist draft mapping to disk (config file).
    """
    global _draft_mapping, _saved_mapping

    ok, msg = _validate_mapping_no_dupes(_draft_mapping)
    if not ok:
        return jsonify({"ok": False, "error": msg}), 409

    _save_mapping(_draft_mapping)
    _saved_mapping = dict(_draft_mapping)

    return jsonify({
        "ok": True,
        "save_enabled": False,
        "reset_enabled": False,
    })


@app.post("/api/command")
def api_command():
    """
    Body: { "location": "<key>", "angle_deg": <int> }

    Uses draft mapping (so you can test before saving).
    """
    global _draft_mapping

    data = request.get_json(force=True)
    loc_key = str(data.get("location", "")).strip()
    if loc_key not in _draft_mapping:
        return jsonify({"ok": False, "error": f"Unknown location '{loc_key}'"}), 400

    ch = _draft_mapping.get(loc_key)
    if ch is None:
        return jsonify({"ok": False, "error": f"Location '{loc_key}' is unassigned"}), 409

    try:
        angle = int(data.get("angle_deg"))
    except Exception:
        return jsonify({"ok": False, "error": "angle_deg must be an integer"}), 400

    angle = _clamp_int(angle, ANGLE_MIN_DEG, ANGLE_MAX_DEG)

    # Command the hardware
    pca.set_channel_angle_deg(ch, angle)

    return jsonify({"ok": True})


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # LAN access: host="0.0.0.0"
    app.run(host="0.0.0.0", port=5000, debug=True)
