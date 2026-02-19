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
    keys = timeline.location_keys
    sparse_input = any(len(t.targets) < len(keys) for t in timeline.ticks[1:])
    rebuilt_ticks: List[TimelineTick] = []

    for i, tick in enumerate(timeline.ticks):
        cur = states[i]
        if i == 0:
            targets = dict(cur)
        elif sparse_input:
            prev = states[i - 1]
            targets = {k: int(cur[k]) for k in keys if int(cur[k]) != int(prev[k])}
        else:
            targets = {k: int(cur[k]) for k in keys}
        rebuilt_ticks.append(TimelineTick(tick=tick.tick, t_ms=tick.t_ms, targets=targets, meta=tick.meta))

    return CompiledTimeline(
        program_id=timeline.program_id,
        tick_ms=timeline.tick_ms,
        duration_ms=timeline.duration_ms,
        location_keys=list(timeline.location_keys),
        ticks=rebuilt_ticks,
    )


def apply_global_max_delta_per_tick(timeline: CompiledTimeline, max_delta_per_tick: int) -> CompiledTimeline:
    max_delta = max(1, int(max_delta_per_tick))
    states = _timeline_to_dense_states(timeline)
    keys = timeline.location_keys

    for i in range(1, len(states)):
        prev = states[i - 1]
        cur = states[i]
        nxt = dict(cur)
        for k in keys:
            delta = int(cur[k]) - int(prev[k])
            if delta > max_delta:
                nxt[k] = int(prev[k]) + max_delta
            elif delta < -max_delta:
                nxt[k] = int(prev[k]) - max_delta
            else:
                nxt[k] = int(cur[k])
            nxt[k] = clamp_int(int(nxt[k]), 0, 270)
        states[i] = nxt

    return _dense_states_to_timeline(timeline, states)


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
