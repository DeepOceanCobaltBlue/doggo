from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

from flask import Flask, jsonify, request, send_from_directory

from app_shared.program_common import file_exists, read_json_file, summarize_timeline, validate_location_order
from hardware.pca9685 import PCA9685
from motion_core import (
    build_config_state_view,
    clamp_int,
    CommandRunner,
    HardwareRuntimeLock,
    ProgramRuntimeEngine,
    SafetyPipeline,
)
from sim_core import sim_store


# -----------------------------
# Hub settings (tunable)
# -----------------------------
HEARTBEAT_TIMEOUT_MS_DEFAULT = 0.0
HEARTBEAT_TIMEOUT_MS_MIN = 0.0
HEARTBEAT_TIMEOUT_MS_MAX = 60_000.0

I2C_BUS = 1
I2C_ADDR = 0x40

CONFIG_DIR = Path("config")
CONFIG_FILE = CONFIG_DIR / "config_file.json"
LOCATION_ORDER_FILE = CONFIG_DIR / "location_order.json"
DYN_VARS_FILE = CONFIG_DIR / "dynamic_limits.json"

# Hub UI
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
HUB_PAGE = "hub_page.html"


# -----------------------------
# In-memory runtime state
# -----------------------------
@dataclass
class HubConfig:
    position_order: List[str]
    locations: Dict[str, Dict[str, Any]]
    dynamic_limits: Dict[str, Any]


class HubState:
    def __init__(self) -> None:
        self.enabled: bool = False
        self.running: bool = False

        self.config: Optional[HubConfig] = None
        self.compiled_timeline: Optional[Any] = None

        self.heartbeat_timeout_ms: float = float(HEARTBEAT_TIMEOUT_MS_DEFAULT)
        self.stop_on_clamp: bool = False
        self.last_heartbeat_at: float = 0.0

        self.pca: Optional[PCA9685] = None
        self.hardware_error: Optional[str] = None

        self.stop_event = threading.Event()

        self.packet_idx: int = 0
        self.last_warning: Optional[str] = None
        self.last_run_reason: Optional[str] = None

        self.telemetry: List[Dict[str, Any]] = []

        self.normal_angles: Dict[str, int] = {}

        self._thread: Optional[threading.Thread] = None
        self._runtime_engine: Optional[ProgramRuntimeEngine] = None
        self._hardware_lock = HardwareRuntimeLock()
        self._lock = threading.Lock()

    # ---------- Hardware ----------
    def _ensure_hardware(self) -> bool:
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
        with self._lock:
            if not file_exists(LOCATION_ORDER_FILE):
                return False, f"Missing {LOCATION_ORDER_FILE}"
            if not file_exists(CONFIG_FILE):
                return False, f"Missing {CONFIG_FILE}"

            try:
                lo = read_json_file(LOCATION_ORDER_FILE)
                order = lo.get("position_order", None)
                ok, cleaned, err = validate_location_order(order)
                if not ok:
                    return False, err

                cfg = read_json_file(CONFIG_FILE)
                if int(cfg.get("version", 0)) != 2:
                    return False, "config_file.json version must be 2"

                locs = cfg.get("locations", {})
                if not isinstance(locs, dict):
                    return False, "config_file.json locations must be an object"

                for k in cleaned:
                    if k not in locs:
                        return False, f"config_file.json missing location key '{k}'"

                dyn = sim_store.load_dyn_vars(DYN_VARS_FILE, location_keys=cleaned)
                self.config = HubConfig(position_order=cleaned, locations=locs, dynamic_limits=dyn)

                # Initialize runtime state on first load.
                if not self.normal_angles:
                    self.normal_angles = {k: 135 for k in cleaned}
                else:
                    self.normal_angles = {k: clamp_int(int(self.normal_angles.get(k, 135)), 0, 270) for k in cleaned}
                return True, "ok"
            except Exception as e:
                return False, str(e)

    def stop(self) -> None:
        with self._lock:
            self.enabled = False
            self.stop_event.set()
            if self._runtime_engine is not None:
                self._runtime_engine.request_stop()

        self._ensure_hardware()
        self._disable_outputs_best_effort()
        self._hardware_lock.release()

        with self._lock:
            self.running = False

    def enable(self) -> Tuple[bool, str]:
        with self._lock:
            self.stop_event.clear()
            self.enabled = True
            self.last_warning = None
            self.last_heartbeat_at = time.monotonic()

        ok = self._ensure_hardware()
        if not ok:
            return True, "enabled (hardware unavailable; dev mode)"
        return True, "enabled"

    def heartbeat(self) -> None:
        with self._lock:
            self.last_heartbeat_at = time.monotonic()

    def _should_stop_runtime(self, heartbeat_timeout_ms: float) -> bool:
        if self.stop_event.is_set():
            return True
        if heartbeat_timeout_ms <= 0.0:
            return False
        with self._lock:
            return (time.monotonic() - float(self.last_heartbeat_at)) * 1000.0 >= heartbeat_timeout_ms

    def start(self) -> Tuple[bool, str]:
        with self._lock:
            if not self.enabled:
                return False, "Not enabled"
            if self.config is None:
                return False, "Config not loaded"
            if self.running:
                return False, "Already running"
            if self.compiled_timeline is None:
                return False, "No compiled timeline loaded (use programming app)"

            self.running = True
            self.packet_idx = 0
            self.last_warning = None
            self.last_run_reason = None
            self.stop_event.clear()
            self.last_heartbeat_at = time.monotonic()

        ok_lock, msg_lock = self._hardware_lock.acquire("hub_app")
        if not ok_lock:
            with self._lock:
                self.running = False
                self.last_warning = msg_lock
            return False, msg_lock

        with self._lock:
            self._thread = threading.Thread(target=self._run_loop, name="doggo_hub_loop", daemon=True)
            self._thread.start()
            return True, "started"

    # ---------- Execution loop ----------
    def _run_loop(self) -> None:
        with self._lock:
            cfg = self.config
            timeline = self.compiled_timeline
            heartbeat_timeout_ms = float(self.heartbeat_timeout_ms)
            stop_on_clamp = bool(self.stop_on_clamp)

        if cfg is None or timeline is None:
            with self._lock:
                self.running = False
            return

        self._ensure_hardware()

        try:
            cfg_view = build_config_state_view(
                draft_cfg={"locations": cfg.locations},
                dynamic_limits=cfg.dynamic_limits,
                location_keys=cfg.position_order,
            )

            safety = SafetyPipeline(
                dynamic_limits=cfg_view.dynamic_limits,
                servo_limits_by_location=cfg_view.servo_limits_by_location,
            )
            runner = CommandRunner(safety)
            engine = ProgramRuntimeEngine(runner)

            with self._lock:
                self._runtime_engine = engine

            result = engine.run(
                timeline,
                state_name="normal",
                output_target="hardware",
                dynamic_limits=cfg_view.dynamic_limits,
                servo_limits_by_location=cfg_view.servo_limits_by_location,
                channel_by_location=cfg_view.channel_by_location,
                hardware=self.pca,
                state_by_name={"normal": dict(self.normal_angles)},
                stop_on_clamp=stop_on_clamp,
                realtime=True,
                stop_check=lambda: self._should_stop_runtime(heartbeat_timeout_ms),
            )

            with self._lock:
                self.running = False
                self.packet_idx = int(result.packets_executed)
                self.last_run_reason = result.reason
                self.normal_angles = dict(result.final_state_by_name.get("normal", self.normal_angles))
                self.telemetry.extend(
                    {
                        "tick": t.tick,
                        "t_ms": t.t_ms,
                        "commands": t.commands,
                        "clamped_count": t.clamped_count,
                        "errors": list(t.errors),
                    }
                    for t in result.telemetry
                )
                # Keep telemetry bounded.
                if len(self.telemetry) > 2000:
                    self.telemetry = self.telemetry[-2000:]

        except Exception as e:
            with self._lock:
                self.running = False
                self.last_run_reason = "runtime_exception"
                self.last_warning = str(e)
        finally:
            if self.stop_event.is_set():
                self._disable_outputs_best_effort()
            self._hardware_lock.release()
            with self._lock:
                self._runtime_engine = None

    # ---------- Status ----------
    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "running": self.running,
                "config_loaded": self.config is not None,
                "compiled_loaded": self.compiled_timeline is not None,
                "compiled_summary": summarize_timeline(self.compiled_timeline) if self.compiled_timeline is not None else None,
                "heartbeat_timeout_ms": self.heartbeat_timeout_ms,
                "stop_on_clamp": self.stop_on_clamp,
                "packet_idx": self.packet_idx,
                "last_warning": self.last_warning,
                "last_run_reason": self.last_run_reason,
                "hardware_lock_owner": self._hardware_lock.owner,
                "hardware_available": self.pca is not None,
                "hardware_error": self.hardware_error,
                "telemetry_count": len(self.telemetry),
            }

    def telemetry_tail(self, n: int) -> List[Dict[str, Any]]:
        with self._lock:
            n = max(1, min(1000, int(n)))
            return list(self.telemetry[-n:])


# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)
state = HubState()


@app.get("/")
def hub_index():
    if STATIC_DIR.exists() and (STATIC_DIR / HUB_PAGE).exists():
        return send_from_directory(str(STATIC_DIR), HUB_PAGE)
    return jsonify({"message": "hub running (UI not installed yet)", "status": state.status()})


@app.get("/api/status")
def api_status():
    return jsonify(state.status())


@app.get("/api/telemetry")
def api_telemetry():
    try:
        tail = int(request.args.get("tail", 50))
    except Exception:
        tail = 50
    return jsonify({"ok": True, "telemetry": state.telemetry_tail(tail)})


@app.post("/api/heartbeat")
def api_heartbeat():
    state.heartbeat()
    return jsonify({"ok": True, "status": state.status()})


@app.post("/api/load_config")
def api_load_config():
    ok, msg = state.load_config_from_disk()
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


@app.post("/api/estop")
def api_estop():
    state.stop()
    return jsonify({"ok": True, "message": "estop", "status": state.status()})


@app.post("/api/settings")
def api_settings():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400

    updates: Dict[str, float] = {}
    bool_updates: Dict[str, bool] = {}

    if "heartbeat_timeout_ms" in data:
        try:
            heartbeat_timeout_ms = float(data["heartbeat_timeout_ms"])
        except Exception:
            return jsonify({"ok": False, "error": "heartbeat_timeout_ms must be a number"}), 400
        if not (HEARTBEAT_TIMEOUT_MS_MIN <= heartbeat_timeout_ms <= HEARTBEAT_TIMEOUT_MS_MAX):
            return jsonify(
                {"ok": False, "error": f"heartbeat_timeout_ms out of range {HEARTBEAT_TIMEOUT_MS_MIN}..{HEARTBEAT_TIMEOUT_MS_MAX}"}
            ), 400
        updates["heartbeat_timeout_ms"] = heartbeat_timeout_ms

    if "stop_on_clamp" in data:
        bool_updates["stop_on_clamp"] = bool(data["stop_on_clamp"])

    with state._lock:
        if "heartbeat_timeout_ms" in updates:
            state.heartbeat_timeout_ms = updates["heartbeat_timeout_ms"]
        if "stop_on_clamp" in bool_updates:
            state.stop_on_clamp = bool_updates["stop_on_clamp"]
    return jsonify({"ok": True, "status": state.status()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
