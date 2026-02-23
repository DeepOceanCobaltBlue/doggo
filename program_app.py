from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass
from datetime import datetime, UTC
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
    CommandRunner,
    ProgramRuntimeEngine,
    SafetyPipeline,
    apply_global_max_delta_per_tick,
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
EXPORTS_DIR = Path("exports")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
PROGRAM_PAGE = "program_page.html"

LOCATIONS = DOGGO_LOCATIONS
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


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
        self.compiled_base_timeline: Optional[Any] = None
        self.compiled_timeline: Optional[Any] = None
        self.compiled_dense_states: List[Dict[str, int]] = []

        self.normal_angles: Dict[str, int] = {}
        self.sim_angles: Dict[str, int] = {}
        self.sim_tick_idx: int = 0
        self.sim_runtime_running: bool = False
        self.sim_runtime_reason: Optional[str] = None
        self.sim_runtime_loop: bool = True
        self.sim_runtime_speed_scale: float = 2.5
        self._runtime_thread: Optional[threading.Thread] = None
        self._runtime_engine: Optional[ProgramRuntimeEngine] = None
        self._runtime_stop_event = threading.Event()

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
            if self.sim_runtime_running:
                return False, "Playback is running"
            locs = list(self.config.position_order)

        ok, spec, msg = normalize_and_validate_program_spec(raw_program, location_keys=locs)
        if not ok or spec is None:
            return False, msg

        with self._lock:
            self.program_spec = spec
            self.program_name = spec.program_id
            self.compiled_base_timeline = None
            self.compiled_timeline = None
            self.compiled_dense_states = []
            self.sim_tick_idx = 0
            self.sim_runtime_reason = None
            return True, "ok"

    def _build_dense_states(self, timeline: Any, start_state: Dict[str, int]) -> List[Dict[str, int]]:
        dense_states: List[Dict[str, int]] = []
        cur = {k: int(v) for k, v in start_state.items()}
        for tick in timeline.ticks:
            for key, value in tick.targets.items():
                cur[key] = int(value)
            dense_states.append(dict(cur))
        return dense_states

    def _set_playback_timeline(self, timeline: Any, *, reset_sim: bool = True) -> None:
        self.compiled_timeline = timeline
        self.compiled_dense_states = self._build_dense_states(timeline, self.normal_angles)
        if reset_sim:
            self.sim_tick_idx = 0
            if self.compiled_dense_states:
                self.sim_angles = dict(self.compiled_dense_states[0])

    def apply_gait_profile(
        self,
        *,
        max_delta_per_tick: int,
        ease_in_frames: int = 2,
        ease_out_frames: int = 2,
    ) -> Tuple[bool, str]:
        with self._lock:
            if self.sim_runtime_running:
                return False, "Playback is running"
            if self.compiled_base_timeline is None:
                return False, "No compiled base timeline"
            base = self.compiled_base_timeline
            if max_delta_per_tick < 1:
                return False, "max_delta_per_tick must be >= 1"
            if ease_in_frames < 1 or ease_out_frames < 1:
                return False, "ease_in_frames/ease_out_frames must be >= 1"

            out = apply_global_max_delta_per_tick(
                base,
                int(max_delta_per_tick),
                ease_in_frames=int(ease_in_frames),
                ease_out_frames=int(ease_out_frames),
            )
            self._set_playback_timeline(out, reset_sim=True)
            return True, "ok"

    def compile_program(
        self,
        *,
        sparse_targets: bool = True,
        max_delta_per_tick: Optional[int] = None,
        gait_ease_in_frames: int = 2,
        gait_ease_out_frames: int = 2,
    ) -> Tuple[bool, str]:
        with self._lock:
            if self.config is None:
                return False, "Config not loaded"
            if self.program_spec is None:
                return False, "Program not loaded"
            if self.sim_runtime_running:
                return False, "Playback is running"

            base_timeline = compile_timeline(
                self.program_spec,
                location_keys=self.config.position_order,
                start_state=self.normal_angles,
                sparse_targets=bool(sparse_targets),
                max_delta_per_tick=None,
                apply_slew_limits=True,
                gait_ease_in_frames=int(gait_ease_in_frames),
                gait_ease_out_frames=int(gait_ease_out_frames),
            )
            self.compiled_base_timeline = base_timeline
            self._set_playback_timeline(base_timeline, reset_sim=True)
            self.sim_runtime_reason = None

        if max_delta_per_tick is not None:
            return self.apply_gait_profile(
                max_delta_per_tick=int(max_delta_per_tick),
                ease_in_frames=int(gait_ease_in_frames),
                ease_out_frames=int(gait_ease_out_frames),
            )
        return True, "ok"

    def _runtime_tick_callback(self, tick: int, _t_ms: int, state: Dict[str, int]) -> None:
        with self._lock:
            self.sim_tick_idx = max(0, int(tick))
            self.sim_angles = {k: clamp_int(int(v), 0, 270) for k, v in state.items()}

    def _run_sim_loop(self, *, loop: bool, speed_scale: float) -> None:
        while True:
            with self._lock:
                if self.config is None or self.compiled_timeline is None:
                    self.sim_runtime_running = False
                    self.sim_runtime_reason = "missing_config_or_timeline"
                    self._runtime_engine = None
                    return
                cfg = self.config
                timeline = self.compiled_timeline
                start_state = (
                    dict(self.compiled_dense_states[0])
                    if self.compiled_dense_states
                    else {k: 135 for k in cfg.position_order}
                )

            safety = SafetyPipeline(dynamic_limits=cfg.dynamic_limits, servo_limits_by_location=cfg.servo_limits_by_location)
            runner = CommandRunner(safety)
            engine = ProgramRuntimeEngine(runner)
            with self._lock:
                self._runtime_engine = engine
                self.sim_tick_idx = 0
                self.sim_angles = dict(start_state)

            result = engine.run(
                timeline,
                state_name="test",
                output_target="sim",
                dynamic_limits=cfg.dynamic_limits,
                servo_limits_by_location=cfg.servo_limits_by_location,
                channel_by_location={k: None for k in cfg.position_order},
                hardware=None,
                state_by_name={"test": start_state},
                stop_on_clamp=False,
                realtime=True,
                realtime_scale=max(0.25, float(speed_scale)),
                stop_check=lambda: self._runtime_stop_event.is_set(),
                tick_callback=self._runtime_tick_callback,
            )

            with self._lock:
                self.sim_runtime_reason = str(result.reason)
                final_test = result.final_state_by_name.get("test", {})
                if final_test:
                    self.sim_angles = {k: clamp_int(int(v), 0, 270) for k, v in final_test.items()}
                if self.compiled_dense_states:
                    max_idx = len(self.compiled_dense_states) - 1
                    self.sim_tick_idx = max(0, min(int(self.sim_tick_idx), max_idx))
                self._runtime_engine = None

            if self._runtime_stop_event.is_set():
                break
            if not loop:
                break
            if result.reason != "completed":
                break

        with self._lock:
            self.sim_runtime_running = False
            self._runtime_engine = None

    def start_sim_playback(self, *, loop: bool = True, speed_scale: Optional[float] = None) -> Tuple[bool, str]:
        with self._lock:
            if self.config is None:
                return False, "Config not loaded"
            if self.compiled_timeline is None or not self.compiled_dense_states:
                return False, "Program not compiled"
            if self.sim_runtime_running:
                return False, "Playback already running"
            self.sim_runtime_running = True
            self.sim_runtime_reason = None
            self.sim_runtime_loop = bool(loop)
            if speed_scale is None:
                speed = float(self.sim_runtime_speed_scale)
            else:
                speed = float(speed_scale)
            self.sim_runtime_speed_scale = max(0.25, speed)
            self._runtime_stop_event.clear()
            self._runtime_thread = threading.Thread(
                target=self._run_sim_loop,
                kwargs={"loop": bool(loop), "speed_scale": float(self.sim_runtime_speed_scale)},
                name="program_app_sim_loop",
                daemon=True,
            )
            self._runtime_thread.start()
            return True, "ok"

    def stop_sim_playback(self) -> Tuple[bool, str]:
        with self._lock:
            was_running = bool(self.sim_runtime_running)
            self._runtime_stop_event.set()
            if self._runtime_engine is not None:
                self._runtime_engine.request_stop()
            if not was_running:
                return True, "ok"
            return True, "stopping"

    def sim_playback_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "running": bool(self.sim_runtime_running),
                "loop": bool(self.sim_runtime_loop),
                "speed_scale": float(self.sim_runtime_speed_scale),
                "reason": self.sim_runtime_reason,
                "tick": int(self.sim_tick_idx),
            }

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
            if self.sim_runtime_running:
                return False, {"ok": False, "error": "Playback is running"}, 409
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

    def export_program(self) -> Dict[str, Any]:
        with self._lock:
            if self.compiled_timeline is None:
                return {"ok": False, "error": "Program not compiled"}

            timeline = self.compiled_timeline
            tick_ms = int(timeline.tick_ms)
            base_name = str(self.program_name or timeline.program_id or "program").strip() or "program"
            export_program_id = f"{base_name}_produced"
            steps: List[Dict[str, Any]] = []
            location_keys = list(timeline.location_keys)
            dense_states = list(self.compiled_dense_states)
            for idx, tick in enumerate(timeline.ticks):
                full_state: Dict[str, int] = {}
                if idx < len(dense_states):
                    full_state = {str(k): int(v) for k, v in dense_states[idx].items()}
                else:
                    full_state = {str(k): clamp_int(int(self.normal_angles.get(k, 135)), 0, 270) for k in location_keys}
                commands = [
                    {
                        "location": str(loc),
                        "target_angle": int(angle),
                        "duration_ms": tick_ms,
                        "easing": "linear",
                    }
                    for loc, angle in sorted(full_state.items())
                ]
                steps.append(
                    {
                        "step_id": str(tick.meta.step_id or f"t{int(tick.tick):04d}"),
                        "commands": commands,
                    }
                )

            return {
                "ok": True,
                "program": {
                    "program_id": export_program_id,
                    "tick_ms": tick_ms,
                    "steps": steps,
                },
                "timeline_events": [e.to_dict() for e in self.timeline_events],
                "summary": summarize_timeline(timeline),
            }

    @staticmethod
    def _sanitize_export_filename(name: str) -> str:
        base = Path(str(name).strip()).name
        if not base:
            return ""
        if not base.lower().endswith(".json"):
            base = f"{base}.json"
        return base

    def list_export_files(self) -> Dict[str, Any]:
        try:
            files: List[Dict[str, Any]] = []
            for p in EXPORTS_DIR.glob("*.json"):
                st = p.stat()
                files.append(
                    {
                        "name": p.name,
                        "size_bytes": int(st.st_size),
                        "modified_utc": datetime.fromtimestamp(st.st_mtime, UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                    }
                )
            files.sort(key=lambda x: x["modified_utc"], reverse=True)
            return {"ok": True, "files": files}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def save_export_program(self, filename: Optional[str] = None) -> Dict[str, Any]:
        out = self.export_program()
        if not out.get("ok"):
            return out
        program = out.get("program", {})
        timeline_events = out.get("timeline_events", [])
        raw_name = "" if filename is None else str(filename).strip()
        if raw_name:
            safe_name = self._sanitize_export_filename(raw_name)
        else:
            # Timestamp default for quick exports when no explicit name is provided.
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            safe_name = self._sanitize_export_filename(f"export_{ts}.json")
        if not safe_name:
            return {"ok": False, "error": "invalid filename"}
        out_path = EXPORTS_DIR / safe_name
        try:
            payload = {
                "program": program,
                "timeline_events": timeline_events,
                "exported_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "filename": safe_name, "path": str(out_path), "summary": out.get("summary")}

    def import_export_program(self, filename: str) -> Tuple[bool, str]:
        safe_name = self._sanitize_export_filename(filename)
        if not safe_name:
            return False, "filename is required"
        p = EXPORTS_DIR / safe_name
        if not p.exists() or not p.is_file():
            return False, f"file not found: {safe_name}"
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            return False, f"failed to read json: {e}"
        return self.import_export_payload(raw)

    def import_export_payload(self, raw: Any) -> Tuple[bool, str]:
        import_events: Optional[List[Dict[str, Any]]] = None
        program_payload = raw
        if isinstance(raw, dict) and isinstance(raw.get("program"), dict):
            program_payload = raw["program"]
            ev = raw.get("timeline_events", None)
            if isinstance(ev, list):
                import_events = ev
        if not isinstance(program_payload, dict):
            return False, "program file must be a JSON object"
        ok, msg = self.load_program_json(program_payload)
        if not ok:
            return False, msg

        restored_events: List[TimelineEvent] = []
        if import_events is not None:
            for item in import_events:
                if not isinstance(item, dict):
                    return False, "timeline_events must contain objects"
                ok_ev, cleaned, msg_ev = self._validate_timeline_event_fields(
                    side=item.get("side"),
                    joint_key=item.get("joint_key"),
                    start_frame=item.get("start_frame"),
                    end_frame=item.get("end_frame"),
                    angle_deg=item.get("angle_deg"),
                )
                if not ok_ev:
                    return False, f"invalid timeline event in import: {msg_ev}"
                restored_events.append(TimelineEvent(id=0, **cleaned))

            # Ensure no same-joint overlap in imported events.
            for i, e in enumerate(restored_events):
                for j in range(i):
                    prev = restored_events[j]
                    if e.side != prev.side or e.joint_key != prev.joint_key:
                        continue
                    disjoint = e.end_frame < prev.start_frame or e.start_frame > prev.end_frame
                    if not disjoint:
                        return False, "imported timeline events contain overlaps on the same joint"

        with self._lock:
            if import_events is None:
                self.timeline_events = []
                self._next_event_id = 1
            else:
                self.timeline_events = [
                    TimelineEvent(
                        id=i + 1,
                        side=e.side,
                        joint_key=e.joint_key,
                        start_frame=e.start_frame,
                        end_frame=e.end_frame,
                        angle_deg=e.angle_deg,
                    )
                    for i, e in enumerate(restored_events)
                ]
                self._next_event_id = len(self.timeline_events) + 1
        return True, "ok"

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config_loaded": self.config is not None,
                "program_loaded": self.program_spec is not None,
                "program_name": self.program_name,
                "compiled_base_loaded": self.compiled_base_timeline is not None,
                "compiled_base_summary": summarize_timeline(self.compiled_base_timeline) if self.compiled_base_timeline is not None else None,
                "compiled_loaded": self.compiled_timeline is not None,
                "compiled_summary": summarize_timeline(self.compiled_timeline) if self.compiled_timeline is not None else None,
                "sim_tick_idx": int(self.sim_tick_idx),
                "sim_runtime_running": bool(self.sim_runtime_running),
                "sim_runtime_reason": self.sim_runtime_reason,
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
        gait_ease_in_frames: int = 2,
        gait_ease_out_frames: int = 2,
    ) -> Tuple[bool, str]:
        with self._lock:
            if self.config is None:
                return False, "Config not loaded"
            events = list(self.timeline_events)
            location_keys = list(self.config.position_order)
            baseline_targets = {str(k): clamp_int(int(self.normal_angles.get(k, 135)), 0, 270) for k in location_keys}

        if not events:
            return False, "No timeline events"

        tick_ms_i = max(1, int(tick_ms))
        events_sorted = sorted(events, key=lambda e: int(e.id))
        max_frame = max(int(e.end_frame) for e in events_sorted)

        events_by_joint: Dict[str, List[TimelineEvent]] = {str(k): [] for k in location_keys}
        for ev in events_sorted:
            jk = str(ev.joint_key)
            if jk in events_by_joint:
                events_by_joint[jk].append(ev)

        joint_targets_by_frame: Dict[str, List[int]] = {}
        for joint_key in location_keys:
            key = str(joint_key)
            joint_events = sorted(
                events_by_joint.get(key, []),
                key=lambda e: (int(e.start_frame), int(e.end_frame), int(e.id)),
            )
            baseline = int(baseline_targets.get(key, 135))
            frame_values = [baseline] * (max_frame + 1)
            cursor = 0
            current = baseline

            for ev in joint_events:
                start_f = int(ev.start_frame)
                end_f = int(ev.end_frame)
                target = clamp_int(int(ev.angle_deg), 0, 270)

                while cursor < start_f and cursor <= max_frame:
                    frame_values[cursor] = current
                    cursor += 1

                span = max(1, end_f - start_f + 1)
                start_val = current
                for i in range(span):
                    frame_idx = start_f + i
                    if frame_idx > max_frame:
                        break
                    # Interpolate over the event window so movement spans the full event.
                    progress = float(i + 1) / float(span)
                    interp = int(round(start_val + (target - start_val) * progress))
                    frame_values[frame_idx] = clamp_int(interp, 0, 270)
                current = target
                cursor = max(cursor, end_f + 1)

            while cursor <= max_frame:
                frame_values[cursor] = current
                cursor += 1

            joint_targets_by_frame[key] = frame_values

        steps: List[Dict[str, Any]] = []
        for frame in range(max_frame + 1):
            commands = [
                {
                    "location": k,
                    "target_angle": int(joint_targets_by_frame[str(k)][frame]),
                    "duration_ms": tick_ms_i,
                    "easing": "linear",
                }
                for k in sorted(location_keys)
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
        ok_comp, msg_comp = self.compile_program(
            sparse_targets=sparse_targets,
            max_delta_per_tick=max_delta_per_tick,
            gait_ease_in_frames=gait_ease_in_frames,
            gait_ease_out_frames=gait_ease_out_frames,
        )
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


@app.get("/api/sim_play/status")
def api_sim_play_status():
    return jsonify(state.sim_playback_status())


@app.post("/api/sim_play/start")
def api_sim_play_start():
    data = request.get_json(force=True) if request.data else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
    loop = bool(data.get("loop", True))
    speed_scale_raw = data.get("speed_scale", None)
    speed_scale = None
    if speed_scale_raw is not None:
        try:
            speed_scale = float(speed_scale_raw)
        except Exception:
            return jsonify({"ok": False, "error": "speed_scale must be numeric"}), 400
        if speed_scale < 0.25:
            return jsonify({"ok": False, "error": "speed_scale must be >= 0.25"}), 400
    ok, msg = state.start_sim_playback(loop=loop, speed_scale=speed_scale)
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), (200 if ok else 400)


@app.post("/api/sim_play/stop")
def api_sim_play_stop():
    ok, msg = state.stop_sim_playback()
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), (200 if ok else 400)


@app.get("/api/program_preview")
def api_program_preview():
    try:
        count = int(request.args.get("count", 200))
    except Exception:
        count = 200
    out = state.program_preview(count)
    return jsonify(out), (200 if out.get("ok") else 400)


@app.get("/api/program_export")
def api_program_export():
    out = state.export_program()
    return jsonify(out), (200 if out.get("ok") else 400)


@app.post("/api/program_export/save")
def api_program_export_save():
    data = request.get_json(force=True) if request.data else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
    filename = data.get("filename", None)
    if filename is not None:
        filename = str(filename)
    out = state.save_export_program(filename=filename)
    return jsonify(out), (200 if out.get("ok") else 400)


@app.get("/api/exports")
def api_exports_list():
    out = state.list_export_files()
    return jsonify(out), (200 if out.get("ok") else 400)


@app.post("/api/program_import")
def api_program_import():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
    filename = str(data.get("filename", "")).strip()
    if not filename:
        return jsonify({"ok": False, "error": "filename is required"}), 400
    ok, msg = state.import_export_program(filename)
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), (200 if ok else 400)


@app.post("/api/program_import_payload")
def api_program_import_payload():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
    ok, msg = state.import_export_payload(data)
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), (200 if ok else 400)


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

    try:
        ease_in_frames = int(data.get("ease_in_frames", 2))
        ease_out_frames = int(data.get("ease_out_frames", 2))
    except Exception:
        return jsonify({"ok": False, "error": "ease_in_frames/ease_out_frames must be integers"}), 400
    if ease_in_frames < 1:
        return jsonify({"ok": False, "error": "ease_in_frames must be >= 1"}), 400
    if ease_out_frames < 1:
        return jsonify({"ok": False, "error": "ease_out_frames must be >= 1"}), 400

    ok, msg = state.compile_program(
        sparse_targets=sparse_targets,
        max_delta_per_tick=max_delta,
        gait_ease_in_frames=ease_in_frames,
        gait_ease_out_frames=ease_out_frames,
    )
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


@app.post("/api/gait/apply")
def api_gait_apply():
    data = request.get_json(force=True) if request.data else {}
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400

    max_delta_raw = data.get("max_delta_per_tick", None)
    if max_delta_raw is None:
        return jsonify({"ok": False, "error": "max_delta_per_tick is required"}), 400
    try:
        max_delta = int(max_delta_raw)
    except Exception:
        return jsonify({"ok": False, "error": "max_delta_per_tick must be an integer"}), 400
    if max_delta < 1:
        return jsonify({"ok": False, "error": "max_delta_per_tick must be >= 1"}), 400

    try:
        ease_in_frames = int(data.get("ease_in_frames", 2))
        ease_out_frames = int(data.get("ease_out_frames", 2))
    except Exception:
        return jsonify({"ok": False, "error": "ease_in_frames/ease_out_frames must be integers"}), 400
    if ease_in_frames < 1:
        return jsonify({"ok": False, "error": "ease_in_frames must be >= 1"}), 400
    if ease_out_frames < 1:
        return jsonify({"ok": False, "error": "ease_out_frames must be >= 1"}), 400

    ok, msg = state.apply_gait_profile(
        max_delta_per_tick=max_delta,
        ease_in_frames=ease_in_frames,
        ease_out_frames=ease_out_frames,
    )
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), (200 if ok else 400)


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

    try:
        ease_in_frames = int(data.get("ease_in_frames", 2))
        ease_out_frames = int(data.get("ease_out_frames", 2))
    except Exception:
        return jsonify({"ok": False, "error": "ease_in_frames/ease_out_frames must be integers"}), 400
    if ease_in_frames < 1:
        return jsonify({"ok": False, "error": "ease_in_frames must be >= 1"}), 400
    if ease_out_frames < 1:
        return jsonify({"ok": False, "error": "ease_out_frames must be >= 1"}), 400

    ok, msg = state.compile_timeline_events(
        tick_ms=tick_ms,
        sparse_targets=sparse_targets,
        max_delta_per_tick=max_delta,
        gait_ease_in_frames=ease_in_frames,
        gait_ease_out_frames=ease_out_frames,
    )
    code = 200 if ok else 400
    return jsonify({"ok": ok, "message": msg, "status": state.status()}), code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
