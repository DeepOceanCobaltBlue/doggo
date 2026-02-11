from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class LoadedProgram:
    """
    Ragged event list program.

    Arrays are parallel (same length N):
      t[i]     = frame index (>= 0)
      loc[i]   = servo index (0..7) into shared location_order.json
      angle[i] = angle command (0..270)

    num_frames is derived unless provided.
    """
    t: np.ndarray       # int64/uint32 etc
    loc: np.ndarray     # uint8 etc
    angle: np.ndarray   # uint16 etc
    num_frames: int


def load_program_npz(path: str | Path) -> Tuple[Optional[LoadedProgram], str]:
    """
    Returns (program, error_message). If ok, error_message == "ok".
    Minimal checks only:
      - arrays exist
      - lengths match
      - t is non-decreasing
      - loc in 0..7
      - angle in 0..270
      - frame 0 contains all 8 loc indices at least once (dense initial pose)
    """
    p = Path(path)

    if not p.exists():
        return None, f"Program file not found: {p}"

    try:
        with np.load(p, allow_pickle=False) as z:
            if "t" not in z or "loc" not in z or "angle" not in z:
                return None, "NPZ must contain arrays: t, loc, angle"

            t = z["t"]
            loc = z["loc"]
            angle = z["angle"]

            # Optional override (not required): num_frames
            num_frames = int(z["num_frames"]) if "num_frames" in z else -1

    except Exception as e:
        return None, f"Failed to read NPZ: {e}"

    # Coerce to 1D arrays
    try:
        t = np.asarray(t).reshape(-1)
        loc = np.asarray(loc).reshape(-1)
        angle = np.asarray(angle).reshape(-1)
    except Exception as e:
        return None, f"NPZ arrays must be 1D-compatible: {e}"

    n = int(t.shape[0])
    if loc.shape[0] != n or angle.shape[0] != n:
        return None, "t, loc, angle must have the same length"

    if n == 0:
        return None, "Program has zero events"

    # Check monotonic non-decreasing t
    # (writer should sort; hub assumes grouped by frame for pointer-walk)
    try:
        dt = np.diff(t.astype(np.int64, copy=False))
        if np.any(dt < 0):
            return None, "t must be sorted non-decreasing (grouped by frame)"
    except Exception as e:
        return None, f"Failed checking t monotonicity: {e}"

    # Bounds checks
    try:
        loc_i = loc.astype(np.int64, copy=False)
        angle_i = angle.astype(np.int64, copy=False)
        if np.any(loc_i < 0) or np.any(loc_i > 7):
            return None, "loc must be in 0..7"
        if np.any(angle_i < 0) or np.any(angle_i > 270):
            return None, "angle must be in 0..270"
    except Exception as e:
        return None, f"Failed checking loc/angle bounds: {e}"

    # Derive num_frames if missing
    if num_frames < 1:
        try:
            num_frames = int(np.max(t.astype(np.int64, copy=False)) + 1)
        except Exception as e:
            return None, f"Failed deriving num_frames: {e}"

    if num_frames < 1:
        return None, "num_frames derived invalid"

    # Dense frame 0 requirement: loc set must include all 0..7 at least once for t == 0
    try:
        t0_mask = (t.astype(np.int64, copy=False) == 0)
        loc0 = set(loc_i[t0_mask].tolist())
        if loc0 != set(range(8)):
            missing = sorted(set(range(8)) - loc0)
            extra = sorted(loc0 - set(range(8)))
            return None, f"Frame 0 must define all 8 servos exactly at least once. Missing={missing}, extra={extra}"
    except Exception as e:
        return None, f"Failed validating dense frame 0: {e}"

    prog = LoadedProgram(t=t, loc=loc, angle=angle, num_frames=int(num_frames))
    return prog, "ok"
