from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

from smbus2 import SMBus

# PCA9685 registers
_MODE1 = 0x00
_MODE2 = 0x01
_PRESCALE = 0xFE
_LED0_ON_L = 0x06

# MODE1 bits
_RESTART = 0x80
_SLEEP = 0x10
_AI = 0x20  # Auto-Increment

# MODE2 bits
_OUTDRV = 0x04  # Totem pole (recommended for servos)

ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 270


@dataclass(frozen=True)
class ServoPulseRange:
    """Pulse range used for mapping logical degrees -> pulse width."""
    min_us: int = 500   
    max_us: int = 2500


@dataclass(frozen=True)
class ServoLimits:
    """
    Per-servo clamp and direction settings, in logical degree-space (0..270).
    - deg_min/deg_max clamp the logical commanded angle
    - invert flips output direction after logical clamping
    """
    deg_min: int = 0
    deg_max: int = 270
    invert: bool = False


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def resolve_logical_and_physical_angle(angle_deg: int, limits: ServoLimits) -> tuple[int, int]:
    """
    Resolve a requested logical command into:
      (logical_clamped_angle, physical_output_angle)

    Contract:
    - Clamp always happens in logical space (invert-agnostic).
    - Invert is applied only to physical output.
    """
    requested = _clamp_int(int(angle_deg), ANGLE_MIN_DEG, ANGLE_MAX_DEG)

    deg_min = _clamp_int(int(limits.deg_min), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    deg_max = _clamp_int(int(limits.deg_max), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    if deg_max < deg_min:
        deg_min, deg_max = deg_max, deg_min

    logical = _clamp_int(requested, deg_min, deg_max)
    physical = (ANGLE_MAX_DEG - logical) if bool(limits.invert) else logical
    physical = _clamp_int(physical, ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    return logical, physical


class PCA9685:
    """
    Minimal PCA9685 driver with:
    - init @ 50Hz (locked)
    - set_channel_pulse_us(channel, pulse_us)
    - set_channel_angle_deg(channel, angle_deg, limits=...) mapping angle -> pulse range
    - disable_channel_output / disable_all_outputs (stop pulses entirely)
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        address: int = 0x40,
        pulse_range: Optional[ServoPulseRange] = None,
        default_limits: Optional[ServoLimits] = None,
    ):
        self.i2c_bus = i2c_bus
        self.address = address
        self.pulse_range = pulse_range or ServoPulseRange()

        # Locked frequency
        self.freq_hz = 50

        # Default per-servo limits (can override per call)
        self.default_limits = default_limits or ServoLimits()

        # Optional: store per-channel limits centrally (0..15)
        self.channel_limits: Dict[int, ServoLimits] = {}

        self._bus = SMBus(self.i2c_bus)
        self._init_device()

    # ---------- Public API ----------

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass

    def set_limits_for_channel(self, channel: int, limits: ServoLimits) -> None:
        """Persist per-channel limits inside the driver."""
        channel = _clamp_int(int(channel), 0, 15)
        lo = _clamp_int(int(limits.deg_min), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        hi = _clamp_int(int(limits.deg_max), ANGLE_MIN_DEG, ANGLE_MAX_DEG)
        if hi < lo:
            lo, hi = hi, lo
        self.channel_limits[channel] = ServoLimits(deg_min=lo, deg_max=hi, invert=bool(limits.invert))

    def set_channel_angle_deg(self, channel: int, angle_deg: int, limits: Optional[ServoLimits] = None) -> None:
        """
        Resolve logical command -> physical output angle, then map to pulse_us and write PWM.

        IMPORTANT:
        - Mapping is always global (0..270). Limits do NOT rescale mapping.
        - Clamp is logical and invert-agnostic.
        - Invert is applied only at the final output angle.
        """
        channel = _clamp_int(int(channel), 0, 15)

        eff_limits = limits or self.channel_limits.get(channel) or self.default_limits
        _, angle_deg = resolve_logical_and_physical_angle(int(angle_deg), eff_limits)

        # Global mapping (stable)
        t = angle_deg / 270.0
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

        pulse_us = int(round(
            self.pulse_range.min_us + t * (self.pulse_range.max_us - self.pulse_range.min_us)
        ))
        self.set_channel_pulse_us(channel, pulse_us)


    def set_channel_pulse_us(self, channel: int, pulse_us: int) -> None:
        channel = _clamp_int(int(channel), 0, 15)
        pulse_us = int(pulse_us)

        # Convert pulse_us -> 12-bit counts (0..4095) for a 20ms frame at 50Hz
        counts = int(round((pulse_us / 1_000_000.0) * self.freq_hz * 4096))
        counts = _clamp_int(counts, 0, 4095)

        # For servos, ON can be 0, OFF is counts
        self._set_pwm(channel, on_count=0, off_count=counts)

    # ---- Stop pulses entirely (disable outputs) ----

    def disable_channel_output(self, channel: int) -> None:
        """
        Stops pulses entirely on a channel using the PCA9685 'FULL OFF' bit.
        Servo will go limp (no holding).
        """
        channel = _clamp_int(int(channel), 0, 15)
        base = _LED0_ON_L + 4 * channel

        # FULL OFF is bit 4 of LEDn_OFF_H (bit 12 overall). Set it to 1.
        # Also clear FULL ON (bit 4 of LEDn_ON_H).
        self._write_u8(base + 0, 0x00)  # ON_L
        self._write_u8(base + 1, 0x00)  # ON_H (FULL ON cleared)
        self._write_u8(base + 2, 0x00)  # OFF_L
        self._write_u8(base + 3, 0x10)  # OFF_H with FULL OFF bit set

    def disable_all_outputs(self) -> None:
        """Stops pulses on all 16 channels."""
        for ch in range(16):
            self.disable_channel_output(ch)

    # ---------- Device init + low-level ----------

    def _init_device(self) -> None:
        # MODE2: totem pole output
        self._write_u8(_MODE2, _OUTDRV)

        # MODE1: auto-increment enabled, awake
        mode1 = self._read_u8(_MODE1)
        mode1 = (mode1 & ~_SLEEP) | _AI
        self._write_u8(_MODE1, mode1)

        # Set frequency to 50Hz
        self._set_pwm_freq_locked_50hz()

    def _set_pwm_freq_locked_50hz(self) -> None:
        """
        Sets PCA9685 prescale for 50Hz.

        Formula (datasheet):
          prescale = round(osc_clock / (4096 * freq)) - 1
        with osc_clock ~ 25MHz (typical internal oscillator).
        """
        osc_clock = 25_000_000.0
        prescale = int(round((osc_clock / (4096.0 * self.freq_hz)) - 1.0))
        prescale = _clamp_int(prescale, 3, 255)

        old_mode = self._read_u8(_MODE1)
        sleep_mode = (old_mode & 0x7F) | _SLEEP  # sleep to set prescale
        self._write_u8(_MODE1, sleep_mode)
        self._write_u8(_PRESCALE, prescale)
        self._write_u8(_MODE1, old_mode)
        # restart
        self._write_u8(_MODE1, old_mode | _RESTART)

    def _set_pwm(self, channel: int, on_count: int, off_count: int) -> None:
        base = _LED0_ON_L + 4 * channel
        on_count = _clamp_int(int(on_count), 0, 4095)
        off_count = _clamp_int(int(off_count), 0, 4095)

        # Note: writing OFF_H without FULL OFF bit implicitly re-enables output for that channel.
        self._write_u8(base + 0, on_count & 0xFF)
        self._write_u8(base + 1, (on_count >> 8) & 0x0F)  # keep FULL ON bit cleared
        self._write_u8(base + 2, off_count & 0xFF)
        self._write_u8(base + 3, (off_count >> 8) & 0x0F)  # keep FULL OFF bit cleared

    def _write_u8(self, reg: int, val: int) -> None:
        self._bus.write_byte_data(self.address, reg, val & 0xFF)

    def _read_u8(self, reg: int) -> int:
        return int(self._bus.read_byte_data(self.address, reg))
