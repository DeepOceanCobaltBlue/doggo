from __future__ import annotations

from typing import Dict, List, Mapping, Optional

from .math_utils import clamp_int
from .program_models import CompiledTimeline, ProgramSpec, TimelineTick

def _timeline_to_dense_states(timeline: CompiledTimeline) -> List[Dict[str, int]]:
    keys = timeline.location_keys
    current = {k: 135 for k in keys}
    if timeline.ticks:
        for k, v in timeline.ticks[0].targets.items():
            current[k] = clamp_int(int(v), 0, 270)

    states: List[Dict[str, int]] = [dict(current)]
    for tick in timeline.ticks[1:]:
        for k, v in tick.targets.items():
            current[k] = clamp_int(int(v), 0, 270)
        states.append(dict(current))
    return states


def _dense_states_to_timeline(timeline: CompiledTimeline, states: List[Dict[str, int]]) -> CompiledTimeline:
    if len(states) != len(timeline.ticks):
        raise ValueError("states length must match timeline ticks length")

    metas = [t.meta for t in timeline.ticks]
    return _build_timeline_from_dense_states(
        timeline,
        states,
        metas,
        sparse_input=any(len(t.targets) < len(timeline.location_keys) for t in timeline.ticks[1:]),
    )


def _build_timeline_from_dense_states(
    timeline: CompiledTimeline,
    states: List[Dict[str, int]],
    metas: List[object],
    *,
    sparse_input: bool,
) -> CompiledTimeline:
    keys = timeline.location_keys
    rebuilt_ticks: List[TimelineTick] = []

    for i, cur in enumerate(states):
        if i == 0:
            targets = dict(cur)
        elif sparse_input:
            prev = states[i - 1]
            targets = {k: int(cur[k]) for k in keys if int(cur[k]) != int(prev[k])}
        else:
            targets = {k: int(cur[k]) for k in keys}

        rebuilt_ticks.append(
            TimelineTick(
                tick=i,
                t_ms=i * int(timeline.tick_ms),
                targets=targets,
                meta=metas[i],
            )
        )

    return CompiledTimeline(
        program_id=timeline.program_id,
        tick_ms=timeline.tick_ms,
        duration_ms=max(0, (len(rebuilt_ticks) - 1) * int(timeline.tick_ms)),
        location_keys=list(timeline.location_keys),
        ticks=rebuilt_ticks,
    )


def _ramp_template(max_delta: int, frames: int) -> List[int]:
    if frames <= 0:
        return []
    out: List[int] = []
    prev = 0
    for i in range(1, frames + 1):
        val = int(round((float(i) / float(frames + 1)) * float(max_delta)))
        val = clamp_int(val, 1, max_delta)
        if val < prev:
            val = prev
        out.append(int(val))
        prev = int(val)
    return out


def build_capped_eased_path(
    *,
    start: int,
    target: int,
    max_delta_per_tick: int,
    ease_in_frames: int = 2,
    ease_out_frames: int = 2,
) -> List[int]:
    start_i = clamp_int(int(start), 0, 270)
    target_i = clamp_int(int(target), 0, 270)
    max_delta = max(1, int(max_delta_per_tick))
    if start_i == target_i:
        return []

    sign = 1 if target_i > start_i else -1
    distance = abs(int(target_i) - int(start_i))

    in_frames = max(0, int(ease_in_frames))
    out_frames = max(0, int(ease_out_frames))

    while True:
        ramp_up = _ramp_template(max_delta, in_frames)
        ramp_down = list(reversed(_ramp_template(max_delta, out_frames)))
        base_sum = sum(ramp_up) + sum(ramp_down)
        if base_sum <= distance or (in_frames == 0 and out_frames == 0):
            break
        if out_frames >= in_frames and out_frames > 0:
            out_frames -= 1
        elif in_frames > 0:
            in_frames -= 1
        else:
            break

    remaining = distance - (sum(ramp_up) + sum(ramp_down))
    cruise = max(0, remaining // max_delta)
    rem = max(0, remaining % max_delta)
    deltas = list(ramp_up) + ([max_delta] * cruise) + list(ramp_down)

    if rem > 0:
        in_len = len(ramp_up)
        cruise_len = int(cruise)
        out_start = in_len + cruise_len
        order = list(range(out_start, len(deltas))) + list(range(in_len - 1, -1, -1))
        for idx in order:
            if rem <= 0:
                break
            headroom = max_delta - int(deltas[idx])
            if headroom <= 0:
                continue
            add = min(headroom, rem)
            deltas[idx] = int(deltas[idx]) + int(add)
            rem -= int(add)
        if rem > 0:
            bridge = min(max_delta, rem)
            deltas.insert(out_start, int(bridge))
            rem -= int(bridge)
        while rem > 0:
            add = min(max_delta, rem)
            deltas.insert(in_len, int(add))
            rem -= int(add)

    if not deltas:
        deltas = [min(max_delta, distance)]

    angles: List[int] = []
    cur = int(start_i)
    for d in deltas:
        step = max(0, int(d))
        cur = clamp_int(int(cur) + (sign * step), 0, 270)
        angles.append(int(cur))

    if int(angles[-1]) != int(target_i):
        angles.append(int(target_i))

    return angles


def apply_global_max_delta_per_tick(
    timeline: CompiledTimeline,
    max_delta_per_tick: int,
    *,
    ease_in_frames: int = 2,
    ease_out_frames: int = 2,
) -> CompiledTimeline:
    states = _timeline_to_dense_states(timeline)
    keys = timeline.location_keys
    if not states or not timeline.ticks:
        return timeline

    sparse_input = any(len(t.targets) < len(keys) for t in timeline.ticks[1:])
    out_states: List[Dict[str, int]] = [dict(states[0])]
    out_metas: List[object] = [timeline.ticks[0].meta]

    for i in range(1, len(states)):
        prev = out_states[-1]
        desired = states[i]
        by_key: Dict[str, List[int]] = {}
        step_len = 0
        for k in keys:
            path = build_capped_eased_path(
                start=int(prev[k]),
                target=int(desired[k]),
                max_delta_per_tick=max_delta_per_tick,
                ease_in_frames=ease_in_frames,
                ease_out_frames=ease_out_frames,
            )
            by_key[k] = path
            step_len = max(step_len, len(path))
        if step_len == 0:
            # Preserve no-op frame intervals so source-frame timing does not collapse.
            out_states.append(dict(out_states[-1]))
            out_metas.append(timeline.ticks[i].meta)
            continue

        for j in range(step_len):
            frame = dict(out_states[-1])
            for k in keys:
                path = by_key[k]
                if not path:
                    continue
                idx = min(j, len(path) - 1)
                frame[k] = int(path[idx])
            out_states.append(frame)
            out_metas.append(timeline.ticks[i].meta)

    return _build_timeline_from_dense_states(
        timeline,
        out_states,
        out_metas,
        sparse_input=sparse_input,
    )


def apply_command_slew_limits(timeline: CompiledTimeline, program: ProgramSpec) -> CompiledTimeline:
    states = _timeline_to_dense_states(timeline)
    keys = timeline.location_keys

    # step_index -> location -> max_deg_per_sec
    limits_by_step: Dict[int, Dict[str, Optional[float]]] = {}
    for i, step in enumerate(program.steps):
        limits_by_step[i] = {cmd.location: cmd.max_deg_per_sec for cmd in step.commands}

    budget_by_loc: Dict[str, float] = {k: 0.0 for k in keys}
    tick_dt_s = float(timeline.tick_ms) / 1000.0

    for i in range(1, len(states)):
        prev = states[i - 1]
        desired = states[i]
        meta = timeline.ticks[i].meta
        step_lims = limits_by_step.get(int(meta.step_index), {})
        nxt = dict(desired)

        for k in keys:
            lim = step_lims.get(k, None)
            if lim is None:
                budget_by_loc[k] = 0.0
                nxt[k] = int(desired[k])
                continue

            budget_by_loc[k] += max(0.0, float(lim)) * tick_dt_s
            allowed = int(budget_by_loc[k])
            needed = int(desired[k]) - int(prev[k])
            step = 0
            if allowed > 0 and needed != 0:
                step = min(abs(needed), allowed) * (1 if needed > 0 else -1)
                budget_by_loc[k] -= abs(step)
            nxt[k] = clamp_int(int(prev[k]) + int(step), 0, 270)

        states[i] = nxt

    return _dense_states_to_timeline(timeline, states)
