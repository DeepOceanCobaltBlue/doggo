from __future__ import annotations

from typing import Dict, List

from .math_utils import clamp_int
from .program_models import CommandPacket, CompiledTimeline

def packetize_timeline(
    timeline: CompiledTimeline,
    *,
    sparse_targets: bool = True,
    include_initial: bool = False,
    include_empty: bool = True,
) -> List[CommandPacket]:
    keys = timeline.location_keys
    current: Dict[str, int] = {k: 135 for k in keys}
    states: List[Dict[str, int]] = []

    for tick in timeline.ticks:
        for k, v in tick.targets.items():
            current[k] = clamp_int(int(v), 0, 270)
        states.append(dict(current))

    packets: List[CommandPacket] = []
    start_idx = 0 if include_initial else 1

    for i, tick in enumerate(timeline.ticks):
        if i < start_idx:
            continue

        if sparse_targets:
            if i == 0:
                targets = {k: int(states[i][k]) for k in keys}
            else:
                targets = {k: int(states[i][k]) for k in keys if int(states[i][k]) != int(states[i - 1][k])}
        else:
            targets = {k: int(states[i][k]) for k in keys}

        if not include_empty and not targets:
            continue
        packets.append(CommandPacket(tick=tick.tick, t_ms=tick.t_ms, targets=targets))

    return packets
