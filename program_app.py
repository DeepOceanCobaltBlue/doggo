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


@dataclass
class TimelineEvent:
    id: int
    side: str
    joint_key: str
    start_frame: int
    end_frame: int
    angle_deg: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "side": str(self.side),
            "joint_key": str(self.joint_key),
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
            "angle_deg": int(self.angle_deg),
        }


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

        self.timeline_events: List[TimelineEvent] = []
        self._next_event_id: int = 1

        self._lock = threading.Lock()

    @staticmethod
    def _timeline_joint_keys_for_side(side: str) -> List[str]:
        if side == "right":
            return ["front_right_hip", "front_right_knee", "rear_right_hip", "rear_right_knee"]
        return ["front_left_hip", "front_left_knee", "rear_left_hip", "rear_left_knee"]

    def _validate_timeline_event_fields(
        self,
        *,
        side: Any,
        joint_key: Any,
        start_frame: Any,
        end_frame: Any,
        angle_deg: Any,
    ) -> Tuple[bool, Dict[str, Any], str]:
        side_s = str(side).strip().lower()
        if side_s not in ("left", "right"):
            return False, {}, "side must be 'left' or 'right'"

        joint_s = str(joint_key).strip()
        if joint_s not in self._timeline_joint_keys_for_side(side_s):
            return False, {}, f"joint_key '{joint_s}' is invalid for side '{side_s}'"

        try:
            start_i = int(start_frame)
            end_i = int(end_frame)
        except Exception:
            return False, {}, "start_frame/end_frame must be integers"
        if start_i < 0 or end_i < 0:
            return False, {}, "start_frame/end_frame must be >= 0"
        if end_i < start_i:
            return False, {}, "end_frame must be >= start_frame"

        try:
            angle_i = int(angle_deg)
        except Exception:
            return False, {}, "angle_deg must be an integer"
        if not (0 <= angle_i <= 270):
            return False, {}, "angle_deg must be in 0..270"

        return (
            True,
            {
                "side": side_s,
                "joint_key": joint_s,
                "start_frame": start_i,
                "end_frame": end_i,
                "angle_deg": angle_i,
            },
            "ok",
        )

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
                "timeline_event_count": len(self.timeline_events),
            }

    def list_timeline_events(self, *, side: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if side is None:
                events = self.timeline_events
            else:
                side_s = str(side).strip().lower()
                events = [e for e in self.timeline_events if e.side == side_s]
            return [e.to_dict() for e in events]

    def create_timeline_event(self, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        ok, cleaned, msg = self._validate_timeline_event_fields(
            side=payload.get("side"),
            joint_key=payload.get("joint_key"),
            start_frame=payload.get("start_frame"),
            end_frame=payload.get("end_frame"),
            angle_deg=payload.get("angle_deg"),
        )
        if not ok:
            return False, {}, msg

        with self._lock:
            overlap_id = self._find_timeline_overlap_id(
                side=cleaned["side"],
                joint_key=cleaned["joint_key"],
                start_frame=int(cleaned["start_frame"]),
                end_frame=int(cleaned["end_frame"]),
                ignore_event_id=None,
            )
            if overlap_id is not None:
                return False, {}, f"event overlaps existing event #{overlap_id} on {cleaned['joint_key']}"
            event = TimelineEvent(id=self._next_event_id, **cleaned)
            self._next_event_id += 1
            self.timeline_events.append(event)
            return True, event.to_dict(), "ok"

    def update_timeline_event(self, event_id: int, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        with self._lock:
            target: Optional[TimelineEvent] = None
            for e in self.timeline_events:
                if e.id == event_id:
                    target = e
                    break
            if target is None:
                return False, {}, "event not found"

            candidate = {
                "side": payload.get("side", target.side),
                "joint_key": payload.get("joint_key", target.joint_key),
                "start_frame": payload.get("start_frame", target.start_frame),
                "end_frame": payload.get("end_frame", target.end_frame),
                "angle_deg": payload.get("angle_deg", target.angle_deg),
            }

        ok, cleaned, msg = self._validate_timeline_event_fields(**candidate)
        if not ok:
            return False, {}, msg

        with self._lock:
            overlap_id = self._find_timeline_overlap_id(
                side=cleaned["side"],
                joint_key=cleaned["joint_key"],
                start_frame=int(cleaned["start_frame"]),
                end_frame=int(cleaned["end_frame"]),
                ignore_event_id=int(event_id),
            )
            if overlap_id is not None:
                return False, {}, f"event overlaps existing event #{overlap_id} on {cleaned['joint_key']}"
            for i, e in enumerate(self.timeline_events):
                if e.id == event_id:
                    self.timeline_events[i] = TimelineEvent(id=event_id, **cleaned)
                    return True, self.timeline_events[i].to_dict(), "ok"
            return False, {}, "event not found"

    def _find_timeline_overlap_id(
        self,
        *,
        side: str,
        joint_key: str,
        start_frame: int,
        end_frame: int,
        ignore_event_id: Optional[int],
    ) -> Optional[int]:
        for existing in self.timeline_events:
            if ignore_event_id is not None and int(existing.id) == int(ignore_event_id):
                continue
            if existing.side != side or existing.joint_key != joint_key:
                continue
            disjoint = end_frame < int(existing.start_frame) or start_frame > int(existing.end_frame)
            if not disjoint:
                return int(existing.id)
        return None

    def delete_timeline_event(self, event_id: int) -> Tuple[bool, str]:
        with self._lock:
            before = len(self.timeline_events)
            self.timeline_events = [e for e in self.timeline_events if e.id != event_id]
            if len(self.timeline_events) == before:
                return False, "event not found"
            return True, "ok"

    def compile_timeline_events(
        self,
        *,
        tick_ms: int = 20,
        sparse_targets: bool = True,
        max_delta_per_tick: Optional[int] = None,
    ) -> Tuple[bool, str]:
        with self._lock:
            if self.config is None:
                return False, "Config not loaded"
            events = list(self.timeline_events)
            location_keys = list(self.config.position_order)

        if not events:
            return False, "No timeline events"

        tick_ms_i = max(1, int(tick_ms))
        events_sorted = sorted(events, key=lambda e: int(e.id))
        max_frame = max(int(e.end_frame) for e in events_sorted)

        steps: List[Dict[str, Any]] = []
        for frame in range(max_frame + 1):
            targets: Dict[str, int] = {}
            for ev in events_sorted:
                if ev.start_frame <= frame <= ev.end_frame:
                    targets[ev.joint_key] = int(ev.angle_deg)
            if not targets:
                continue
            commands = [
                {
                    "location": k,
                    "target_angle": int(v),
                    "duration_ms": tick_ms_i,
                    "easing": "linear",
                }
                for k, v in sorted(targets.items())
            ]
            steps.append({"step_id": f"f{frame}", "commands": commands})

        if not steps:
            return False, "No active timeline commands to compile"

        raw_program = {
            "program_id": "timeline_events_program",
            "tick_ms": tick_ms_i,
            "steps": steps,
        }

        ok_load, msg_load = self.load_program_json(raw_program)
        if not ok_load:
            return False, msg_load
        ok_comp, msg_comp = self.compile_program(sparse_targets=sparse_targets, max_delta_per_tick=max_delta_per_tick)
        if not ok_comp:
            return False, msg_comp
        return True, "ok"


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


@app.get("/api/timeline/events")
def api_timeline_events_get():
    side = request.args.get("side", None)
    if side is not None:
        side = str(side).strip().lower()
        if side not in ("left", "right"):
            return jsonify({"ok": False, "error": "side must be 'left' or 'right'"}), 400
    events = state.list_timeline_events(side=side)
    return jsonify({"ok": True, "events": events})


@app.post("/api/timeline/events")
def api_timeline_events_post():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
    ok, event, msg = state.create_timeline_event(data)
    code = 200 if ok else 400
    return jsonify({"ok": ok, "event": event if ok else None, "error": None if ok else msg}), code


@app.patch("/api/timeline/events/<int:event_id>")
def api_timeline_events_patch(event_id: int):
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
    ok, event, msg = state.update_timeline_event(event_id, data)
    code = 200 if ok else 404 if msg == "event not found" else 400
    return jsonify({"ok": ok, "event": event if ok else None, "error": None if ok else msg}), code


@app.delete("/api/timeline/events/<int:event_id>")
def api_timeline_events_delete(event_id: int):
    ok, msg = state.delete_timeline_event(event_id)
    code = 200 if ok else 404
    return jsonify({"ok": ok, "error": None if ok else msg}), code


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


@app.post("/api/timeline/compile")
def api_timeline_compile():
    data = request.get_json(force=True) if request.data else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400

    sparse_targets = bool(data.get("sparse_targets", True))
    tick_ms_raw = data.get("tick_ms", 20)
    try:
        tick_ms = int(tick_ms_raw)
    except Exception:
        return jsonify({"ok": False, "error": "tick_ms must be an integer"}), 400
    if tick_ms < 1:
        return jsonify({"ok": False, "error": "tick_ms must be >= 1"}), 400

    max_delta_raw = data.get("max_delta_per_tick", None)
    max_delta = None
    if max_delta_raw is not None:
        try:
            max_delta = int(max_delta_raw)
        except Exception:
            return jsonify({"ok": False, "error": "max_delta_per_tick must be an integer"}), 400
        if max_delta < 1:
            return jsonify({"ok": False, "error": "max_delta_per_tick must be >= 1"}), 400

    ok, msg = state.compile_timeline_events(
        tick_ms=tick_ms,
        sparse_targets=sparse_targets,
        max_delta_per_tick=max_delta,
    )
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
