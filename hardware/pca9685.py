"""
Doggo PCA9685 driver (Raspberry Pi) + angle(0..270) -> pulse_us -> PWM counts.

- Frequency is LOCKED at 50 Hz.
- Channels are 0..15.
- UI never sees pulses; backend uses this driver to command channels.

Requires:
  pip install smbus2
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from smbus2 import SMBus

# PCA9685 registers
_MODE1      = 0x00
_MODE2      = 0x01
_SUBADR1    = 0x02
_SUBADR2    = 0x03
_SUBADR3    = 0x04
_PRESCALE   = 0xFE
_LED0_ON_L  = 0x06

# MODE1 bits
_RESTART = 0x80
_SLEEP   = 0x10
_AI      = 0x20  # Auto-Increment

# MODE2 bits
_OUTDRV  = 0x04  # Totem pole (recommended for servos)


@dataclass(frozen=True)
class ServoPulseRange:
    """Default pulse range used for mapping degrees -> pulse width."""
    min_us: int = 900
    max_us: int = 2100


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


class PCA9685:
    """
    Minimal PCA9685 driver with:
    - init @ 50Hz
    - set_channel_pulse_us(channel, pulse_us)
    - set_channel_angle_deg(channel, angle_deg) mapping 0..270 -> pulse range
    """

    def __init__(
        self,
        i2c_bus: int = 1,
        address: int = 0x40,
        pulse_range: Optional[ServoPulseRange] = None,
    ):
        self.i2c_bus = i2c_bus
        self.address = address
        self.pulse_range = pulse_range or ServoPulseRange()

        # Locked frequency
        self.freq_hz = 50

        self._bus = SMBus(self.i2c_bus)
        self._init_device()

    # ---------- Public API ----------

    def close(self) -> None:
        try:
            self._bus.close()
        except Exception:
            pass

    def set_channel_angle_deg(self, channel: int, angle_deg: int) -> int:
        """
        Maps angle_deg (0..270) to pulse_us using the default pulse_range,
        writes it to the channel, and returns the pulse_us used.
        """
        channel = _clamp_int(int(channel), 0, 15)
        angle_deg = _clamp_int(int(angle_deg), 0, 270)

        # Linear map angle -> pulse_us
        t = angle_deg / 270.0
        pulse_us = int(round(self.pulse_range.min_us + t * (self.pulse_range.max_us - self.pulse_range.min_us)))

        self.set_channel_pulse_us(channel, pulse_us)
        return pulse_us

    def set_channel_pulse_us(self, channel: int, pulse_us: int) -> None:
        """
        Writes a pulse width (microseconds) to a channel at 50Hz.
        """
        channel = _clamp_int(int(channel), 0, 15)
        pulse_us = int(pulse_us)

        # Convert pulse_us -> 12-bit counts (0..4095) for a 20ms frame at 50Hz
        # counts = pulse_seconds * freq_hz * 4096
        counts = int(round((pulse_us / 1_000_000.0) * self.freq_hz * 4096))
        counts = _clamp_int(counts, 0, 4095)

        # For servos, ON can be 0, OFF is counts
        self._set_pwm(channel, on_count=0, off_count=counts)

    # ---------- Device init + low-level ----------

    def _init_device(self) -> None:
        """
        Initialize PCA9685:
        - set MODE2 OUTDRV
        - enable auto-increment
        - set prescale for 50Hz
        """
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
        Sets PCA9685 prescale for exactly 50Hz.

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
        """
        Writes PWM on/off counts to a channel.
        """
        base = _LED0_ON_L + 4 * channel
        on_count = _clamp_int(int(on_count), 0, 4095)
        off_count = _clamp_int(int(off_count), 0, 4095)

        self._write_u8(base + 0, on_count & 0xFF)
        self._write_u8(base + 1, (on_count >> 8) & 0xFF)
        self._write_u8(base + 2, off_count & 0xFF)
        self._write_u8(base + 3, (off_count >> 8) & 0xFF)

    def _write_u8(self, reg: int, val: int) -> None:
        self._bus.write_byte_data(self.address, reg, val & 0xFF)

    def _read_u8(self, reg: int) -> int:
        return int(self._bus.read_byte_data(self.address, reg))


if __name__ == "__main__":
    # Quick manual sanity test (be careful):
    # Moves channel 0 to 0°, 135°, 270°.
    import time

    p = PCA9685(i2c_bus=1, address=0x40)
    try:
        for deg in (0, 135, 270, 135):
            p.set_channel_angle_deg(0, deg)
            time.sleep(0.8)
    finally:
        p.close()
