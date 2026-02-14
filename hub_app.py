from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from flask import Flask, jsonify, request, send_from_directory

from hardware.pca9685 import PCA9685, ServoLimits
from program_loader_npz import load_program_npz, LoadedProgram as NPZProgram


# -----------------------------
# Hub settings (tunable)
# -----------------------------
SLICE_PERIOD_MS_DEFAULT = 40         # you will tune this
OVERRUN_WARN_MS_DEFAULT = 1.0        # you will tune this
SLICE_PERIOD_MS_MIN = 1.0
SLICE_PERIOD_MS_MAX = 5000.0
OVERRUN_WARN_MS_MIN = 0.0
OVERRUN_WARN_MS_MAX = 5000.0

I2C_BUS = 1
I2C_ADDR = 0x40

CONFIG_DIR = Path("config")
CONFIG_FILE = CONFIG_DIR / "config_file.json"
LOCATION_ORDER_FILE = CONFIG_DIR / "location_order.json"

PROGRAMS_DIR = Path("programs")

# Hub UI
BASE_DIR = Path(__file__).resolve().parent
HUB_STATIC_DIR = BASE_DIR / "hub_static"
HUB_PAGE = "hub_page.html"


# -----------------------------
# Helpers
# -----------------------------
def _now() -> float:
    return time.monotonic()


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _file_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _validate_location_order(order: Any) -> Tuple[bool, List[str], str]:
    if not isinstance(order, list):
        return False, [], "location_order.position_order must be a list"
    cleaned = [str(x).strip() for x in order if str(x).strip()]
    if len(cleaned) != 8:
        return False, [], "position_order must contain exactly 8 location keys"
    if len(set(cleaned)) != 8:
        return False, [], "position_order contains duplicate keys"
    return True, cleaned, ""


def _limits_from_cfg_item(item: Dict[str, Any]) -> ServoLimits:
    lim = item.get("limits", {}) if isinstance(item.get("limits", {}), dict) else {}
    deg_min = _clamp_int(int(lim.get("deg_min", 0)), 0, 270)
    deg_max = _clamp_int(int(lim.get("deg_max", 270)), 0, 270)
    if deg_max < deg_min:
        deg_min, deg_max = deg_max, deg_min
    invert = bool(lim.get("invert", False))
    return ServoLimits(deg_min=deg_min, deg_max=deg_max, invert=invert)


def _resolve_program_path(program_name_or_path: str) -> Tuple[Optional[Path], str]:
    """
    Accept either:
      - "default_stance.npz" (looked up under programs/)
      - "programs/default_stance.npz"
      - absolute path to a .npz

    Returns (path, "ok") or (None, error).
    """
    s = (program_name_or_path or "").strip()
    if not s:
        return None, "program must be provided"

    p = Path(s)

    # If it's a bare name, resolve under programs/
    if not p.is_absolute() and p.parent == Path("."):
        p = PROGRAMS_DIR / p

    # If it's a relative path, resolve from cwd
    p = p.resolve()

    if p.suffix.lower() != ".npz":
        return None, "program must be a .npz file"
    if not p.exists():
        return None, f"program not found: {p}"
    return p, "ok"


# -----------------------------
# In-memory runtime state
# -----------------------------
@dataclass
class HubConfig:
    position_order: List[str]                  # length 8
    locations: Dict[str, Dict[str, Any]]       # from config_file.json


class HubState:
    def __init__(self) -> None:
        # Core flags
        self.enabled: bool = False
        self.running: bool = False

        # Config + program
        self.config: Optional[HubConfig] = None
        self.program: Optional[NPZProgram] = None
        self.program_path: Optional[str] = None

        # Timing
        self.slice_period_ms: float = float(SLICE_PERIOD_MS_DEFAULT)
        self.overrun_warn_ms: float = float(OVERRUN_WARN_MS_DEFAULT)

        # Hardware (optional)
        self.pca: Optional[PCA9685] = None
        self.hardware_error: Optional[str] = None

        # Control events
        self.stop_event = threading.Event()

        # Runtime telemetry
        self.frame_idx: int = 0
        self.last_overrun_ms: float = 0.0
        self.overrun_count: int = 0
        self.last_warning: Optional[str] = None

        # Execution thread
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # ---------- Hardware ----------
    def _ensure_hardware(self) -> bool:
        """
        Lazy init hardware. Hub must be able to run without it (dev mode).
        """
        if self.pca is not None:
            return True
        try:
            self.pca = PCA9685(i2c_bus=I2C_BUS, address=I2C_ADDR)
            self.hardware_error = None
            return True
        except Exception as e:
            self.pca = None
            self.hardware_error = str(e)
            return False

    def _disable_outputs_best_effort(self) -> None:
        if self.pca is None:
            return
        try:
            self.pca.disable_all_outputs()
        except Exception as e:
            self.hardware_error = str(e)

    # ---------- Public operations ----------
    def load_config_from_disk(self) -> Tuple[bool, str]:
        """
        Loads:
          - config/location_order.json (position_order list)
          - config/config_file.json (channel + limits)
        """
        with self._lock:
            if not _file_exists(LOCATION_ORDER_FILE):
                return False, f"Missing {LOCATION_ORDER_FILE}"
            if not _file_exists(CONFIG_FILE):
                return False, f"Missing {CONFIG_FILE}"

            try:
                lo = _read_json(LOCATION_ORDER_FILE)
                order = lo.get("position_order", None)
                ok, cleaned, err = _validate_location_order(order)
                if not ok:
                    return False, err

                cfg = _read_json(CONFIG_FILE)
                if int(cfg.get("version", 0)) != 2:
                    return False, "config_file.json version must be 2"

                locs = cfg.get("locations", {})
                if not isinstance(locs, dict):
                    return False, "config_file.json locations must be an object"

                # Ensure keys exist for position_order
                for k in cleaned:
                    if k not in locs:
                        return False, f"config_file.json missing location key '{k}'"

                self.config = HubConfig(position_order=cleaned, locations=locs)
                return True, "ok"
            except Exception as e:
                return False, str(e)

    def load_program(self, program_name_or_path: str) -> Tuple[bool, str]:
        """
        Loads NPZ ragged event list into memory.
        Minimal checks happen here; full validation happens in writer.
        """
        p, msg = _resolve_program_path(program_name_or_path)
        if p is None:
            return False, msg

        prog, err = load_program_npz(p)
        if prog is None:
            return False, err

        with self._lock:
            self.program = prog
            self.program_path = str(p)
            return True, "ok"

    def stop(self) -> None:
        """
        stop = stop pulses entirely and halt playback.
        """
        with self._lock:
            self.enabled = False
            self.stop_event.set()

        # Stop outputs immediately (best effort)
        self._ensure_hardware()
        self._disable_outputs_best_effort()

        with self._lock:
            self.running = False

    def enable(self) -> Tuple[bool, str]:
        """
        enable = allow outputs again, but DO NOT send any motion.
        """
        with self._lock:
            self.stop_event.clear()
            self.enabled = True
            self.last_warning = None

        # Hardware is optional; enable can succeed without hardware (dev mode).
        ok = self._ensure_hardware()
        if not ok:
            return True, "enabled (hardware unavailable; dev mode)"
        return True, "enabled"

    def start(self) -> Tuple[bool, str]:
        """
        Starts program playback in a thread.
        Preconditions:
          - enabled == True
          - config loaded
          - program loaded
        """
        with self._lock:
            if not self.enabled:
                return False, "Not enabled"
            if self.config is None:
                return False, "Config not loaded"
            if self.program is None:
                return False, "Program not loaded"
            if self.running:
                return False, "Already running"

            self.running = True
            self.frame_idx = 0
            self.last_overrun_ms = 0.0
            self.overrun_count = 0
            self.last_warning = None

            self._thread = threading.Thread(target=self._run_loop, name="doggo_hub_loop", daemon=True)
            self._thread.start()
            return True, "started"

    # ---------- Execution loop ----------
    def _run_loop(self) -> None:
        """
        Sparse ragged event list playback, pointer-walk.
        - Frame 0 must be dense (checked by loader)
        - Subsequent frames sparse
        - Never drops frames
        - Deadline scheduling with monotonic clock
        - Overrun warnings when lateness >= overrun_warn_ms
        """
        with self._lock:
            cfg = self.config
            prog = self.program
            slice_period_ms = float(self.slice_period_ms)
            overrun_warn_ms = float(self.overrun_warn_ms)

        if cfg is None or prog is None:
            with self._lock:
                self.running = False
            return

        hw_ok = self._ensure_hardware()

        period_s = max(0.001, slice_period_ms / 1000.0)

        # Targets vector (degrees) by loc index 0..7.
        # Frame 0 is dense (loader-validated), so we can initialize by consuming frame 0 events.
        targets: List[int] = [0] * 8

        # Pointer-walk: arrays are numpy; keep an integer index
        i = 0
        n_events = int(prog.t.shape[0])

        start_t = _now()
        frame = 0
        num_frames = int(prog.num_frames)

        while frame < num_frames:
            if self.stop_event.is_set():
                break

            deadline = start_t + frame * period_s

            updated_locs: List[int] = []
            while i < n_events and int(prog.t[i]) == frame:
                loc_idx = int(prog.loc[i])
                ang = int(prog.angle[i])
                # loc and angle already bounds-checked by loader, but keep safe anyway
                if 0 <= loc_idx <= 7:
                    targets[loc_idx] = _clamp_int(ang, 0, 270)
                    updated_locs.append(loc_idx)
                i += 1

            # Send only updated servos for this frame
            if hw_ok and self.pca is not None and updated_locs:
                for loc_idx in updated_locs:
                    loc_key = cfg.position_order[loc_idx]
                    loc_item = cfg.locations.get(loc_key, {})
                    ch = loc_item.get("channel", None)
                    if ch is None:
                        continue
                    try:
                        ch_i = int(ch)
                    except Exception:
                        continue
                    limits = _limits_from_cfg_item(loc_item)
                    try:
                        self.pca.set_channel_angle_deg(ch_i, int(targets[loc_idx]), limits=limits)
                    except Exception as e:
                        with self._lock:
                            self.hardware_error = str(e)

            # Lateness / overrun detection
            now = _now()
            lateness_ms = max(0.0, (now - deadline) * 1000.0)
            if lateness_ms >= overrun_warn_ms:
                with self._lock:
                    self.last_overrun_ms = lateness_ms
                    self.overrun_count += 1
                    self.last_warning = (
                        f"Overrun: lateness={lateness_ms:.3f}ms >= warn={overrun_warn_ms:.3f}ms. "
                        f"Increase slice_period_ms (currently {slice_period_ms:.1f}ms) or reduce per-frame updates."
                    )

            # Sleep to next deadline (no drops)
            next_deadline = start_t + (frame + 1) * period_s
            sleep_s = next_deadline - _now()
            if sleep_s > 0:
                time.sleep(sleep_s)

            frame += 1
            with self._lock:
                self.frame_idx = frame

        with self._lock:
            self.running = False

        # If stop requested, disable pulses
        if self.stop_event.is_set() and hw_ok:
            self._disable_outputs_best_effort()

    # ---------- Status ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "running": self.running,
                "config_loaded": self.config is not None,
                "program_loaded": self.program is not None,
                "program_path": self.program_path,
                "slice_period_ms": self.slice_period_ms,
                "overrun_warn_ms": self.overrun_warn_ms,
                "frame_idx": self.frame_idx,
                "overrun_count": self.overrun_count,
                "last_overrun_ms": self.last_overrun_ms,
                "last_warning": self.last_warning,
                "hardware_available": self.pca is not None,
                "hardware_error": self.hardware_error,
            }


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
state = HubState()


@app.get("/")
def hub_index():
    if HUB_STATIC_DIR.exists() and (HUB_STATIC_DIR / HUB_PAGE).exists():
        return send_from_directory(str(HUB_STATIC_DIR), HUB_PAGE)
    return jsonify({"message": "hub running (UI not installed yet)", "status": state.status()})


@app.get("/api/status")
def api_status():
    return jsonify(state.status())


@app.post("/api/load_config")
def api_load_config():
    ok, msg = state.load_config_from_disk()
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


@app.post("/api/load_program")
def api_load_program():
    """
    Body:
      { "program": "default_stance.npz" }
    or:
      { "program": "programs/default_stance.npz" }
    """
    data = request.get_json(force=True)
    prog = str(data.get("program", "")).strip()

    ok, msg = state.load_program(prog)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


@app.post("/api/enable")
def api_enable():
    ok, msg = state.enable()
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), (200 if ok else 400)


@app.post("/api/start")
def api_start():
    ok, msg = state.start()
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), (200 if ok else 400)


@app.post("/api/stop")
def api_stop():
    state.stop()
    return jsonify({"ok": True, "message": "stopped", "status": state.status()})


@app.post("/api/settings")
def api_settings():
    """
    Body:
      { "slice_period_ms": <float>, "overrun_warn_ms": <float> }
    """
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400

    updates: Dict[str, float] = {}
    if "slice_period_ms" in data:
        try:
            slice_period_ms = float(data["slice_period_ms"])
        except Exception:
            return jsonify({"ok": False, "error": "slice_period_ms must be a number"}), 400
        if not (SLICE_PERIOD_MS_MIN <= slice_period_ms <= SLICE_PERIOD_MS_MAX):
            return jsonify(
                {"ok": False, "error": f"slice_period_ms out of range {SLICE_PERIOD_MS_MIN}..{SLICE_PERIOD_MS_MAX}"}
            ), 400
        updates["slice_period_ms"] = slice_period_ms

    if "overrun_warn_ms" in data:
        try:
            overrun_warn_ms = float(data["overrun_warn_ms"])
        except Exception:
            return jsonify({"ok": False, "error": "overrun_warn_ms must be a number"}), 400
        if not (OVERRUN_WARN_MS_MIN <= overrun_warn_ms <= OVERRUN_WARN_MS_MAX):
            return jsonify(
                {"ok": False, "error": f"overrun_warn_ms out of range {OVERRUN_WARN_MS_MIN}..{OVERRUN_WARN_MS_MAX}"}
            ), 400
        updates["overrun_warn_ms"] = overrun_warn_ms

    with state._lock:
        if "slice_period_ms" in updates:
            state.slice_period_ms = updates["slice_period_ms"]
        if "overrun_warn_ms" in updates:
            state.overrun_warn_ms = updates["overrun_warn_ms"]
    return jsonify({"ok": True, "status": state.status()})


if __name__ == "__main__":
    PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="0.0.0.0", port=5001, debug=True)
