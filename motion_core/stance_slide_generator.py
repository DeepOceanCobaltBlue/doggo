from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .taskspace_kinematics import leg_descriptor, solve_leg_ik_for_toe, toe_from_leg_angles


def joint_keys_for_side_leg(side: str, leg: str) -> Tuple[str, str]:
    side_s = str(side).strip().lower()
    leg_s = str(leg).strip().lower()
    if side_s == "right":
        return (
            "front_right_hip" if leg_s == "front" else "rear_right_hip",
            "front_right_knee" if leg_s == "front" else "rear_right_knee",
        )
    return (
        "front_left_hip" if leg_s == "front" else "rear_left_hip",
        "front_left_knee" if leg_s == "front" else "rear_left_knee",
    )


def generate_stance_slide_events(
    *,
    dynamic_limits: Dict[str, Any],
    locations_cfg: Dict[str, Dict[str, Any]],
    start_state: Dict[str, int],
    side: str,
    leg: str,
    start_frame: int,
    duration_frames: int,
    slide_dx_mm: float,
    toe_y_lock_mm: Optional[float],
    sample_every_n_frames: int,
) -> Dict[str, Any]:
    side_s = str(side).strip().lower()
    leg_s = str(leg).strip().lower()
    if side_s not in ("left", "right"):
        raise ValueError("side must be left or right")
    if leg_s not in ("front", "rear"):
        raise ValueError("leg must be front or rear")

    start_f = int(start_frame)
    dur_f = max(1, int(duration_frames))
    sample_n = max(1, int(sample_every_n_frames))

    side_vars = dynamic_limits[side_s]
    desc = leg_descriptor(side_s, leg_s, side_vars)
    hip_key, knee_key = joint_keys_for_side_leg(side_s, leg_s)
    hip_loc_cfg = locations_cfg.get(hip_key, {})
    knee_loc_cfg = locations_cfg.get(knee_key, {})

    hip0 = int(start_state.get(hip_key, 135))
    knee0 = int(start_state.get(knee_key, 135))
    toe0 = toe_from_leg_angles(desc, hip0, knee0)
    lock_y = float(toe_y_lock_mm) if toe_y_lock_mm is not None else float(toe0[1])

    # Side-local "forward" axis is +x for left and -x for right.
    axis = 1.0 if side_s == "left" else -1.0

    def _solve_for_slide(slide_dx_used: float) -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        prev_cmd: Optional[Tuple[float, float]] = (float(hip0), float(knee0))
        for i in range(dur_f + 1):
            p = float(i) / float(dur_f)
            target_x = float(toe0[0]) - (axis * float(slide_dx_used) * p)
            target_y = float(lock_y)
            sol = solve_leg_ik_for_toe(
                desc=desc,
                toe_target=(target_x, target_y),
                prev_cmd=prev_cmd,
                hip_loc_cfg=hip_loc_cfg,
                knee_loc_cfg=knee_loc_cfg,
            )
            frames.append(
                {
                    "frame": int(start_f + i),
                    "hip_cmd": int(sol["hip_cmd"]),
                    "knee_cmd": int(sol["knee_cmd"]),
                    "toe_x": float(sol["toe_x"]),
                    "toe_y": float(sol["toe_y"]),
                    "target_toe_x": float(target_x),
                    "target_toe_y": float(target_y),
                    "unreachable": bool(sol["unreachable"]),
                    "clamped": bool(sol["clamped"]),
                }
            )
            prev_cmd = (float(sol["hip_cmd"]), float(sol["knee_cmd"]))
        return frames

    requested_slide_dx = float(slide_dx_mm)
    used_slide_dx = float(slide_dx_mm)
    auto_scaled = False
    scale_attempts = 0
    frame_solutions: List[Dict[str, Any]] = _solve_for_slide(used_slide_dx)
    unreachable_count = sum(1 for s in frame_solutions if bool(s["unreachable"]))
    # Auto-reduce slide distance until all frames are reachable.
    while unreachable_count > 0 and scale_attempts < 10:
        used_slide_dx *= 0.8
        auto_scaled = True
        scale_attempts += 1
        frame_solutions = _solve_for_slide(used_slide_dx)
        unreachable_count = sum(1 for s in frame_solutions if bool(s["unreachable"]))
    if unreachable_count > 0:
        # Guaranteed fallback to a reachable path from current pose.
        used_slide_dx = 0.0
        auto_scaled = True
        scale_attempts += 1
        frame_solutions = _solve_for_slide(used_slide_dx)
        unreachable_count = sum(1 for s in frame_solutions if bool(s["unreachable"]))

    sampled: List[Dict[str, Any]] = []
    for idx, s in enumerate(frame_solutions):
        if (idx % sample_n == 0) or (idx == (len(frame_solutions) - 1)):
            sampled.append(s)

    # Convert sampled angle waypoints into non-overlapping timeline events.
    events: List[Dict[str, Any]] = []
    if sampled:
        prev_frame = int(sampled[0]["frame"])
        prev_hip = int(sampled[0]["hip_cmd"])
        prev_knee = int(sampled[0]["knee_cmd"])
        for s in sampled[1:]:
            cur_frame = int(s["frame"])
            if cur_frame <= prev_frame:
                continue
            cur_hip = int(s["hip_cmd"])
            cur_knee = int(s["knee_cmd"])
            ev_start = prev_frame + 1
            ev_end = cur_frame
            if cur_hip != prev_hip:
                events.append(
                    {
                        "side": side_s,
                        "joint_key": hip_key,
                        "start_frame": int(ev_start),
                        "end_frame": int(ev_end),
                        "angle_deg": int(cur_hip),
                    }
                )
            if cur_knee != prev_knee:
                events.append(
                    {
                        "side": side_s,
                        "joint_key": knee_key,
                        "start_frame": int(ev_start),
                        "end_frame": int(ev_end),
                        "angle_deg": int(cur_knee),
                    }
                )
            prev_frame = cur_frame
            prev_hip = cur_hip
            prev_knee = cur_knee

    clamped_count = sum(1 for s in frame_solutions if bool(s["clamped"]))
    return {
        "hip_joint_key": hip_key,
        "knee_joint_key": knee_key,
        "events": events,
        "samples": frame_solutions,
        "summary": {
            "start_frame": int(start_f),
            "end_frame": int(start_f + dur_f),
            "duration_frames": int(dur_f),
            "sample_every_n_frames": int(sample_n),
            "requested_slide_dx_mm": float(requested_slide_dx),
            "used_slide_dx_mm": float(used_slide_dx),
            "auto_scaled": bool(auto_scaled),
            "scale_attempts": int(scale_attempts),
            "event_count": len(events),
            "unreachable_count": int(unreachable_count),
            "clamped_count": int(clamped_count),
        },
    }
