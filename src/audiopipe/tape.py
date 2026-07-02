from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf

from .segment import EDL, Segment
from .stages.base import Context
from . import io
from .splice import render_edl
from .degrade import degrade, add_hiss, add_flutter


def _character(audio, sr, ctx, *, speed, flutter, hiss):
    """Apply the base tape character (every pass, including cycles:1): steady
    varispeed, then wow/flutter, then hiss last so noise isn't itself warped."""
    if abs(speed - 1.0) >= 1e-6:
        audio = io.resample_to(audio, max(1, round(len(audio) / speed)))
    audio = add_flutter(audio, sr, flutter, ctx.rng_for("tape"))
    audio = add_hiss(audio, hiss, ctx.rng_for("tape"))
    return audio


def _region_frames(region, n: int, sr: int) -> tuple[int, int]:
    """Resolve a [start_sec, end_sec] window of the rendered loop to frame
    bounds. None loops the whole thing."""
    if region is None:
        return 0, n
    if not (isinstance(region, (list, tuple)) and len(region) == 2):
        raise ValueError("tape.region must be [start_sec, end_sec] or null")
    start = max(0, int(region[0] * sr))
    end = min(n, int(region[1] * sr))
    if end <= start:
        raise ValueError(f"tape.region {region} is empty within the "
                         f"{n / sr:.2f}s loop content")
    return start, end


def run_tape(edl: EDL, ctx: Context, cfg: dict, in_path: Path, out_path: Path) -> None:
    """The tape stage: apply physical character (hiss/wear/flutter/varispeed) to
    the rendered master at `in_path`, optionally looping `cycles` copies of a
    `region` of it (render-once, degrade-per-cycle). Writes out_path."""
    cycles = int(cfg["cycles"])
    hiss = float(cfg["hiss"])
    flutter = float(cfg["flutter"])
    speed = float(cfg["speed"])
    reverse = bool(cfg["reverse"])

    sr, ch, full_n = io.info(in_path)
    start, end = _region_frames(cfg.get("region"), full_n, sr)
    original = io.read_frames(in_path, start, end - start, "keep")

    # cycles<=1: a single finishing tape pass (character only, no wear ramp).
    if cycles <= 1:
        cur = _character(original, sr, ctx, speed=speed, flutter=flutter, hiss=hiss)
        if reverse:
            cur = np.ascontiguousarray(cur[::-1])     # play the whole tape backwards
        sf.write(str(out_path), cur, sr)
        edl.record("tape", {"cycles": 1, "speed": speed, "flutter": flutter,
                                 "hiss": hiss, "reverse": reverse,
                                 "region": cfg.get("region")})
        return

    wear_amount = float(cfg["wear"])
    feedback = bool(cfg["feedback"])
    span = max(cycles - 1, 1)

    cycle_segs: list[Segment] = []
    prev = original
    per_cycle_wear = []
    for c in range(cycles):
        wear = wear_amount * (c / span)
        per_cycle_wear.append(wear)
        if feedback:
            # cycle N degrades cycle N-1's *already-worn* output by the ramped
            # wear, so damage compounds (self-feeding) instead of plateauing
            worn = prev if c == 0 else degrade(prev, sr, wear, ctx.rng_for("tape"))
            prev = worn
        else:
            worn = degrade(original, sr, wear, ctx.rng_for("tape"))
        cur = _character(worn, sr, ctx, speed=speed, flutter=flutter, hiss=hiss)
        cpath = ctx.scratch_dir / f"cycle_{c:03d}.wav"
        sf.write(str(cpath), cur, sr)
        cycle_segs.append(Segment(source=cpath, start_frame=0, end_frame=len(cur),
                                  sample_rate=sr, channels=ch, cycle=c,
                                  ops=(f"tape:cycle{c}",)))

    edl.segments = cycle_segs
    seam_edl = EDL(segments=cycle_segs, seed=edl.seed, sample_rate=sr)
    render_edl(seam_edl, out_path, join=cfg["seam"], fade=0.2, channels="keep")
    if reverse:
        y = io.read_frames(out_path, 0, io.frames_of(out_path), "keep")
        sf.write(str(out_path), np.ascontiguousarray(y[::-1]), sr)  # whole tape backwards
    edl.record("tape", {"cycles": cycles, "wear": wear_amount,
                             "feedback": feedback, "seam": cfg["seam"],
                             "speed": speed, "flutter": flutter, "hiss": hiss,
                             "reverse": reverse, "region": cfg.get("region"),
                             "per_cycle_wear": per_cycle_wear})
