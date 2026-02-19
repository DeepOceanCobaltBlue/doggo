from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Tuple

from .program_models import ProgramCommand, ProgramSpec, ProgramStep, VALID_EASING


def _coerce_int(v: Any, fallback: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(fallback)


def normalize_and_validate_program_spec(
    raw: Mapping[str, Any],
    *,
    location_keys: Optional[Iterable[str]] = None,
    default_tick_ms: int = 20,
) -> Tuple[bool, Optional[ProgramSpec], str]:
    if not isinstance(raw, Mapping):
        return False, None, "program must be an object"

    program_id = str(raw.get("program_id", "")).strip()
    if not program_id:
        return False, None, "program_id is required"

    tick_ms = _coerce_int(raw.get("tick_ms", default_tick_ms), default_tick_ms)
    if tick_ms < 1:
        return False, None, "tick_ms must be >= 1"

    allowed_locations = {str(k) for k in location_keys} if location_keys is not None else None

    steps_raw = raw.get("steps", None)
    if not isinstance(steps_raw, list) or not steps_raw:
        return False, None, "steps must be a non-empty array"

    steps: List[ProgramStep] = []
    for i, step_raw in enumerate(steps_raw):
        if not isinstance(step_raw, Mapping):
            return False, None, f"steps[{i}] must be an object"

        step_id = str(step_raw.get("step_id", i + 1)).strip() or str(i + 1)
        advance = str(step_raw.get("advance", "all_complete")).strip() or "all_complete"
        if advance != "all_complete":
            return False, None, f"steps[{i}].advance must be 'all_complete'"

        cmds_raw = step_raw.get("commands", None)
        if not isinstance(cmds_raw, list) or not cmds_raw:
            return False, None, f"steps[{i}].commands must be a non-empty array"

        commands: List[ProgramCommand] = []
        seen_locations = set()
        for j, cmd_raw in enumerate(cmds_raw):
            if not isinstance(cmd_raw, Mapping):
                return False, None, f"steps[{i}].commands[{j}] must be an object"

            loc = str(cmd_raw.get("location", "")).strip()
            if not loc:
                return False, None, f"steps[{i}].commands[{j}].location is required"
            if allowed_locations is not None and loc not in allowed_locations:
                return False, None, f"steps[{i}].commands[{j}].location '{loc}' is unknown"
            if loc in seen_locations:
                return False, None, f"steps[{i}] has duplicate location '{loc}'"
            seen_locations.add(loc)

            target_angle = _coerce_int(cmd_raw.get("target_angle", None), -1)
            if not (0 <= target_angle <= 270):
                return False, None, f"steps[{i}].commands[{j}].target_angle must be in 0..270"

            duration_ms = _coerce_int(cmd_raw.get("duration_ms", None), -1)
            if duration_ms < 1:
                return False, None, f"steps[{i}].commands[{j}].duration_ms must be >= 1"

            easing = str(cmd_raw.get("easing", "linear")).strip().lower() or "linear"
            if easing not in VALID_EASING:
                return False, None, f"steps[{i}].commands[{j}].easing is invalid"

            max_deg_per_sec = cmd_raw.get("max_deg_per_sec", None)
            if max_deg_per_sec is not None:
                try:
                    max_deg_per_sec = float(max_deg_per_sec)
                except Exception:
                    return False, None, f"steps[{i}].commands[{j}].max_deg_per_sec must be numeric"
                if max_deg_per_sec <= 0.0:
                    return False, None, f"steps[{i}].commands[{j}].max_deg_per_sec must be > 0"

            commands.append(
                ProgramCommand(
                    location=loc,
                    target_angle=target_angle,
                    duration_ms=duration_ms,
                    easing=easing,
                    max_deg_per_sec=max_deg_per_sec,
                )
            )

        steps.append(ProgramStep(step_id=step_id, commands=commands, advance=advance))

    return True, ProgramSpec(program_id=program_id, tick_ms=tick_ms, steps=steps), "ok"


def validate_and_build_program_spec(raw: Mapping[str, Any]) -> Tuple[bool, Optional[ProgramSpec], str]:
    """
    Backward-compatible validator entrypoint used by existing tests.
    """
    return normalize_and_validate_program_spec(raw)

