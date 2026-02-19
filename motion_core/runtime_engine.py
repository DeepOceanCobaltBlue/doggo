from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from hardware.pca9685 import ServoLimits

from .command_runner import CommandRunner
from .packetizer import packetize_timeline
from .program_models import CompiledTimeline


@dataclass(frozen=True)
class RuntimeTickTelemetry:
    tick: int
    t_ms: int
    commands: int
    clamped_count: int
    errors: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeRunResult:
    ok: bool
    reason: str
    packets_executed: int
    final_state_by_name: Dict[str, Dict[str, int]]
    telemetry: List[RuntimeTickTelemetry]


class ProgramRuntimeEngine:
    def __init__(self, command_runner: CommandRunner):
        self._command_runner = command_runner
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def reset_stop(self) -> None:
        self._stop_requested = False

    def run(
        self,
        timeline: CompiledTimeline,
        *,
        state_name: str,
        output_target: str,
        dynamic_limits: Mapping[str, Any],
        servo_limits_by_location: Mapping[str, ServoLimits],
        channel_by_location: Mapping[str, Optional[int]],
        hardware: Optional[Any],
        state_by_name: Mapping[str, Mapping[str, int]],
        stop_on_clamp: bool = False,
        realtime: bool = False,
        stop_check: Optional[Callable[[], bool]] = None,
        now_fn=time.monotonic,
        sleep_fn=time.sleep,
    ) -> RuntimeRunResult:
        self.reset_stop()
        state_map: Dict[str, Dict[str, int]] = {str(k): {str(lk): int(v) for lk, v in vals.items()} for k, vals in state_by_name.items()}
        telemetry: List[RuntimeTickTelemetry] = []
        packets = packetize_timeline(timeline, sparse_targets=True, include_initial=False, include_empty=True)

        start_t = now_fn()
        executed = 0
        for packet in packets:
            if self._stop_requested:
                return RuntimeRunResult(
                    ok=False,
                    reason="stopped",
                    packets_executed=executed,
                    final_state_by_name=state_map,
                    telemetry=telemetry,
                )
            if stop_check is not None and bool(stop_check()):
                return RuntimeRunResult(
                    ok=False,
                    reason="stopped",
                    packets_executed=executed,
                    final_state_by_name=state_map,
                    telemetry=telemetry,
                )

            if realtime:
                deadline = start_t + float(packet.t_ms) / 1000.0
                sleep_for = deadline - now_fn()
                if sleep_for > 0:
                    sleep_fn(sleep_for)

            errors: List[str] = []
            clamped_count = 0
            for loc_key, requested in packet.targets.items():
                out = self._command_runner.execute(
                    state_name=state_name,
                    loc_key=loc_key,
                    requested_angle=int(requested),
                    output_target=output_target,
                    dynamic_limits=dynamic_limits,
                    servo_limits_by_location=servo_limits_by_location,
                    state_by_name=state_map,
                    channel_by_location=channel_by_location,
                    hardware=hardware,
                )
                if not out.ok:
                    errors.append(str(out.payload.get("error", "command failed")))
                    telemetry.append(
                        RuntimeTickTelemetry(
                            tick=packet.tick,
                            t_ms=packet.t_ms,
                            commands=len(packet.targets),
                            clamped_count=clamped_count,
                            errors=errors,
                        )
                    )
                    return RuntimeRunResult(
                        ok=False,
                        reason="command_error",
                        packets_executed=executed,
                        final_state_by_name=state_map,
                        telemetry=telemetry,
                    )

                state_map[state_name] = dict(out.updated_state)
                if bool(out.payload.get("clamped", False)):
                    clamped_count += 1
                    if stop_on_clamp:
                        telemetry.append(
                            RuntimeTickTelemetry(
                                tick=packet.tick,
                                t_ms=packet.t_ms,
                                commands=len(packet.targets),
                                clamped_count=clamped_count,
                                errors=errors,
                            )
                        )
                        return RuntimeRunResult(
                            ok=False,
                            reason="clamped",
                            packets_executed=executed,
                            final_state_by_name=state_map,
                            telemetry=telemetry,
                        )

            telemetry.append(
                RuntimeTickTelemetry(
                    tick=packet.tick,
                    t_ms=packet.t_ms,
                    commands=len(packet.targets),
                    clamped_count=clamped_count,
                    errors=errors,
                )
            )
            executed += 1

        return RuntimeRunResult(
            ok=True,
            reason="completed",
            packets_executed=executed,
            final_state_by_name=state_map,
            telemetry=telemetry,
        )
