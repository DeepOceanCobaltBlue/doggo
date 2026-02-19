from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional

from .gait_mutators import apply_command_slew_limits, apply_global_max_delta_per_tick
from .math_utils import clamp_int
from .program_models import CompiledTimeline, TickMeta, TimelineTick, ProgramSpec

def easing_value(name: str, alpha: float) -> float:
    a = max(0.0, min(1.0, float(alpha)))
    if name == "ease_in":
        return a * a
    if name == "ease_out":
        return 1.0 - (1.0 - a) * (1.0 - a)
    if name == "ease_in_out":
        if a < 0.5:
            return 2.0 * a * a
        return 1.0 - pow(-2.0 * a + 2.0, 2.0) / 2.0
    return a


def _interp_angle(start: int, end: int, eased_alpha: float) -> int:
    return clamp_int(int(round(start + (end - start) * eased_alpha)), 0, 270)


def compile_timeline(
    program: ProgramSpec,
    *,
    location_keys: Iterable[str],
    start_state: Mapping[str, int],
    sparse_targets: bool = True,
    max_delta_per_tick: Optional[int] = None,
    apply_slew_limits: bool = True,
) -> CompiledTimeline:
    keys = [str(k) for k in location_keys]
    if program.tick_ms < 1:
        raise ValueError("program.tick_ms must be >= 1")
    if not keys:
        raise ValueError("location_keys must not be empty")

    current_angles: Dict[str, int] = {k: clamp_int(int(start_state.get(k, 135)), 0, 270) for k in keys}
    last_emitted_angles: Dict[str, int] = dict(current_angles)

    ticks: list[TimelineTick] = [
        TimelineTick(
            tick=0,
            t_ms=0,
            targets=dict(current_angles),
            meta=TickMeta(step_id="initial", step_index=-1),
        )
    ]
    tick_idx = 0
    current_t_ms = 0

    for step_index, step in enumerate(program.steps):
        step_duration_ms = max(int(cmd.duration_ms) for cmd in step.commands)
        step_ticks = max(1, int(math.ceil(step_duration_ms / float(program.tick_ms))))
        step_start_angles = dict(current_angles)

        per_cmd_ticks: Dict[str, int] = {}
        for cmd in step.commands:
            per_cmd_ticks[cmd.location] = max(1, int(math.ceil(int(cmd.duration_ms) / float(program.tick_ms))))

        for i in range(1, step_ticks + 1):
            tick_idx += 1
            current_t_ms += int(program.tick_ms)

            frame_angles = dict(current_angles)
            for cmd in step.commands:
                cmd_ticks = per_cmd_ticks[cmd.location]
                alpha = min(1.0, float(i) / float(cmd_ticks))
                eased = easing_value(cmd.easing, alpha)
                start = int(step_start_angles[cmd.location])
                frame_angles[cmd.location] = _interp_angle(start, int(cmd.target_angle), eased)

            current_angles = frame_angles

            if sparse_targets:
                targets = {k: int(v) for k, v in frame_angles.items() if int(v) != int(last_emitted_angles.get(k, 135))}
            else:
                targets = {k: int(v) for k, v in frame_angles.items()}

            ticks.append(
                TimelineTick(
                    tick=tick_idx,
                    t_ms=current_t_ms,
                    targets=targets,
                    meta=TickMeta(step_id=step.step_id, step_index=step_index),
                )
            )
            last_emitted_angles = frame_angles

        for cmd in step.commands:
            current_angles[cmd.location] = clamp_int(int(cmd.target_angle), 0, 270)

    out = CompiledTimeline(
        program_id=program.program_id,
        tick_ms=int(program.tick_ms),
        duration_ms=current_t_ms,
        location_keys=keys,
        ticks=ticks,
    )

    if apply_slew_limits:
        out = apply_command_slew_limits(out, program)
    if max_delta_per_tick is not None:
        out = apply_global_max_delta_per_tick(out, int(max_delta_per_tick))
    return out
