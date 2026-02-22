from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory

from app_shared.program_common import (
    DOGGO_LOCATIONS,
    file_exists,
    read_json_file,
    summarize_timeline,
    validate_location_order,
)
from motion_core import (
    SafetyPipeline,
    build_config_state_view,
    clamp_int,
    compile_timeline,
    normalize_and_validate_program_spec,
)
from sim_core import sim_store


CONFIG_DIR = Path("config")
CONFIG_FILE = CONFIG_DIR / "config_file.json"
LOCATION_ORDER_FILE = CONFIG_DIR / "location_order.json"
DYN_VARS_FILE = CONFIG_DIR / "dynamic_limits.json"

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PROGRAM_PAGE = "program_page.html"

LOCATIONS = DOGGO_LOCATIONS


@dataclass
class ProgramConfig:
    position_order: List[str]
    locations: Dict[str, Dict[str, Any]]
    dynamic_limits: Dict[str, Any]
    channel_by_location: Dict[str, Optional[int]]
    servo_limits_by_location: Dict[str, Any]


class ProgramState:
    def __init__(self) -> None:
        self.config: Optional[ProgramConfig] = None
        self.program_spec: Optional[Any] = None
        self.program_name: Optional[str] = None
        self.compiled_timeline: Optional[Any] = None
        self.compiled_dense_states: List[Dict[str, int]] = []

        self.normal_angles: Dict[str, int] = {}
        self.sim_angles: Dict[str, int] = {}
        self.sim_tick_idx: int = 0

        self._lock = threading.Lock()

    def load_config_from_disk(self) -> Tuple[bool, str]:
        with self._lock:
            if not file_exists(LOCATION_ORDER_FILE):
                return False, f"Missing {LOCATION_ORDER_FILE}"
            if not file_exists(CONFIG_FILE):
                return False, f"Missing {CONFIG_FILE}"

            try:
                lo = read_json_file(LOCATION_ORDER_FILE)
                ok, cleaned, err = validate_location_order(lo.get("position_order", None))
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
                cfg_view = build_config_state_view(
                    draft_cfg={"locations": locs},
                    dynamic_limits=dyn,
                    location_keys=cleaned,
                )
                self.config = ProgramConfig(
                    position_order=cleaned,
                    locations=locs,
                    dynamic_limits=dyn,
                    channel_by_location=cfg_view.channel_by_location,
                    servo_limits_by_location=cfg_view.servo_limits_by_location,
                )

                if not self.normal_angles:
                    self.normal_angles = {k: 135 for k in cleaned}
                else:
                    self.normal_angles = {k: clamp_int(int(self.normal_angles.get(k, 135)), 0, 270) for k in cleaned}

                if not self.sim_angles:
                    self.sim_angles = dict(self.normal_angles)
                else:
                    self.sim_angles = {k: clamp_int(int(self.sim_angles.get(k, 135)), 0, 270) for k in cleaned}

                return True, "ok"
            except Exception as e:
                return False, str(e)

    def load_program_json(self, raw_program: Dict[str, Any]) -> Tuple[bool, str]:
        with self._lock:
            if self.config is None:
                return False, "Config not loaded"
            locs = list(self.config.position_order)

        ok, spec, msg = normalize_and_validate_program_spec(raw_program, location_keys=locs)
        if not ok or spec is None:
            return False, msg

        with self._lock:
            self.program_spec = spec
            self.program_name = spec.program_id
            self.compiled_timeline = None
            self.compiled_dense_states = []
            self.sim_tick_idx = 0
            return True, "ok"

    def _build_dense_states(self, timeline: Any, start_state: Dict[str, int]) -> List[Dict[str, int]]:
        dense_states: List[Dict[str, int]] = []
        cur = {k: int(v) for k, v in start_state.items()}
        for tick in timeline.ticks:
            for key, value in tick.targets.items():
                cur[key] = int(value)
            dense_states.append(dict(cur))
        return dense_states

    def compile_program(self, *, sparse_targets: bool = True, max_delta_per_tick: Optional[int] = None) -> Tuple[bool, str]:
        with self._lock:
            if self.config is None:
                return False, "Config not loaded"
            if self.program_spec is None:
                return False, "Program not loaded"

            timeline = compile_timeline(
                self.program_spec,
                location_keys=self.config.position_order,
                start_state=self.normal_angles,
                sparse_targets=bool(sparse_targets),
                max_delta_per_tick=max_delta_per_tick,
                apply_slew_limits=True,
            )
            self.compiled_timeline = timeline
            self.compiled_dense_states = self._build_dense_states(timeline, self.normal_angles)
            self.sim_tick_idx = 0
            if self.compiled_dense_states:
                self.sim_angles = dict(self.compiled_dense_states[0])
            return True, "ok"

    def collision_snapshot_for_state(self, state_angles: Dict[str, int]) -> Dict[str, Any]:
        with self._lock:
            if self.config is None:
                return {"left": {"collides": False, "details": {}}, "right": {"collides": False, "details": {}}}
            dynamic_limits = self.config.dynamic_limits
            servo_limits = self.config.servo_limits_by_location

        safety = SafetyPipeline(dynamic_limits=dynamic_limits, servo_limits_by_location=servo_limits)
        return safety.collision_snapshot_for_state(state_angles)

    def sim_state(self) -> Tuple[bool, Dict[str, Any], int]:
        with self._lock:
            if self.config is None:
                return False, {"ok": False, "error": "Config not loaded"}, 400

            payload = {
                "ok": True,
                "locations": [{"key": loc.key, "label": loc.label} for loc in LOCATIONS],
                "location_order": list(self.config.position_order),
                "draft": {"locations": self.config.locations},
                "dynamic_limits": self.config.dynamic_limits,
                "hw_angles": dict(self.normal_angles),
                "sim_angles": dict(self.sim_angles),
                "sim_tick_idx": int(self.sim_tick_idx),
            }

        payload["collision_status"] = {
            "normal": self.collision_snapshot_for_state(payload["hw_angles"]),
            "test": self.collision_snapshot_for_state(payload["sim_angles"]),
        }
        return True, payload, 200

    def seek_tick(self, tick: int) -> Tuple[bool, Dict[str, Any], int]:
        with self._lock:
            if self.compiled_timeline is None or not self.compiled_dense_states:
                return False, {"ok": False, "error": "Program not compiled"}, 400

            idx = max(0, min(len(self.compiled_dense_states) - 1, int(tick)))
            self.sim_tick_idx = idx
            self.sim_angles = dict(self.compiled_dense_states[idx])
            sim_angles = dict(self.sim_angles)

        collision = self.collision_snapshot_for_state(sim_angles)
        return (
            True,
            {
                "ok": True,
                "sim_tick_idx": int(self.sim_tick_idx),
                "sim_angles": sim_angles,
                "collision_status": {"test": collision},
            },
            200,
        )

    def program_preview(self, n: int) -> Dict[str, Any]:
        with self._lock:
            n = max(1, min(500, int(n)))
            if self.compiled_timeline is None:
                return {"ok": False, "error": "Program not compiled"}

            ticks = self.compiled_timeline.ticks[:n]
            out_ticks = [
                {
                    "tick": int(t.tick),
                    "t_ms": int(t.t_ms),
                    "targets": dict(t.targets),
                    "meta": {"step_id": t.meta.step_id, "step_index": int(t.meta.step_index)},
                }
                for t in ticks
            ]
            return {"ok": True, "summary": summarize_timeline(self.compiled_timeline), "ticks": out_ticks}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config_loaded": self.config is not None,
                "program_loaded": self.program_spec is not None,
                "program_name": self.program_name,
                "compiled_loaded": self.compiled_timeline is not None,
                "compiled_summary": summarize_timeline(self.compiled_timeline) if self.compiled_timeline is not None else None,
                "sim_tick_idx": int(self.sim_tick_idx),
            }


app = Flask(__name__)
state = ProgramState()


@app.get("/")
def program_index():
    if STATIC_DIR.exists() and (STATIC_DIR / PROGRAM_PAGE).exists():
        return send_from_directory(str(STATIC_DIR), PROGRAM_PAGE)
    return jsonify({"message": "program app running (UI not installed yet)", "status": state.status()})


@app.get("/api/status")
def api_status():
    return jsonify(state.status())


@app.post("/api/load_config")
def api_load_config():
    ok, msg = state.load_config_from_disk()
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


@app.get("/api/sim_state")
def api_sim_state():
    _ok, payload, code = state.sim_state()
    return jsonify(payload), code


@app.post("/api/sim_seek")
def api_sim_seek():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
    try:
        tick = int(data.get("tick", 0))
    except Exception:
        return jsonify({"ok": False, "error": "tick must be an integer"}), 400

    _ok, payload, code = state.seek_tick(tick)
    return jsonify(payload), code


@app.get("/api/program_preview")
def api_program_preview():
    try:
        count = int(request.args.get("count", 200))
    except Exception:
        count = 200
    out = state.program_preview(count)
    return jsonify(out), (200 if out.get("ok") else 400)


@app.post("/api/load_program_json")
def api_load_program_json():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400

    prog = data.get("program", None)
    if not isinstance(prog, dict):
        return jsonify({"ok": False, "error": "program must be an object"}), 400

    ok, msg = state.load_program_json(prog)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


@app.post("/api/load_program")
def api_load_program():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400

    prog = data.get("program", None)
    if not isinstance(prog, dict):
        return jsonify({"ok": False, "error": "program must be an object (NPZ path loading removed)"}), 400

    ok, msg = state.load_program_json(prog)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


@app.post("/api/compile_program")
def api_compile_program():
    data = request.get_json(force=True) if request.data else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400

    sparse_targets = bool(data.get("sparse_targets", True))
    max_delta_raw = data.get("max_delta_per_tick", None)
    max_delta = None
    if max_delta_raw is not None:
        try:
            max_delta = int(max_delta_raw)
        except Exception:
            return jsonify({"ok": False, "error": "max_delta_per_tick must be an integer"}), 400
        if max_delta < 1:
            return jsonify({"ok": False, "error": "max_delta_per_tick must be >= 1"}), 400

    ok, msg = state.compile_program(sparse_targets=sparse_targets, max_delta_per_tick=max_delta)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
