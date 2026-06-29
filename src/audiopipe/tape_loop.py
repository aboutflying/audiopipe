from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf

from .segment import EDL, Segment
from .stages.base import Context
from . import io
from .splice import render_edl
from .degrade import degrade


def _loop_content(edl: EDL, ctx: Context) -> Path:
    """The chain's single rendered loop file. Render it if the chain didn't end
    in splice (so a loop file always exists before degrade)."""
    segs = edl.segments
    if len(segs) == 1 and ctx.scratch_dir in Path(segs[0].source).parents:
        return Path(segs[0].source)
    path = ctx.scratch_dir / "loop.wav"
    render_edl(edl, path, join="cut", smear=0.0, channels=ctx.channels)
    return path


def _region_frames(region, n: int, sr: int) -> tuple[int, int]:
    """Resolve a [start_sec, end_sec] window of the rendered loop to frame
    bounds. None loops the whole thing."""
    if region is None:
        return 0, n
    if not (isinstance(region, (list, tuple)) and len(region) == 2):
        raise ValueError("tape_loop.region must be [start_sec, end_sec] or null")
    start = max(0, int(region[0] * sr))
    end = min(n, int(region[1] * sr))
    if end <= start:
        raise ValueError(f"tape_loop.region {region} is empty within the "
                         f"{n / sr:.2f}s loop content")
    return start, end


def run_tape_loop(edl: EDL, ctx: Context, cfg: dict, out_path: Path) -> None:
    """Post-chain: build `cycles` copies of the rendered loop (optionally just a
    `region` of it), degrade each by its wear (render-once, degrade-per-cycle),
    concatenate with the seam join."""
    cycles = int(cfg["cycles"])
    loop_path = _loop_content(edl, ctx)

    sr, ch, full_n = io.info(loop_path)
    start, end = _region_frames(cfg.get("region"), full_n, sr)
    original = io.read_frames(loop_path, start, end - start, "keep")
    n = len(original)

    if cycles <= 1:
        sf.write(str(out_path), original, sr)
        edl.record("tape_loop", {"cycles": 1, "region": cfg.get("region")})
        return

    evolve = float(cfg["evolve"])
    recursive = bool(cfg["recursive"])
    span = max(cycles - 1, 1)

    cycle_segs: list[Segment] = []
    prev = original
    per_cycle_wear = []
    for c in range(cycles):
        wear = evolve * (c / span)
        per_cycle_wear.append(wear)
        if recursive:
            # cycle N degrades cycle N-1's output by one constant step
            cur = prev.copy() if c == 0 else degrade(prev, sr, evolve / span, ctx.rng)
        else:
            cur = degrade(original, sr, wear, ctx.rng)
        prev = cur
        cpath = ctx.scratch_dir / f"cycle_{c:03d}.wav"
        sf.write(str(cpath), cur, sr)
        cycle_segs.append(Segment(source=cpath, start_frame=0, end_frame=len(cur),
                                  sample_rate=sr, channels=ch, cycle=c,
                                  ops=(f"tape:cycle{c}",)))

    edl.segments = cycle_segs
    seam_edl = EDL(segments=cycle_segs, seed=edl.seed, sample_rate=sr)
    render_edl(seam_edl, out_path, join=cfg["seam"], smear=0.2, channels="keep")
    edl.record("tape_loop", {"cycles": cycles, "evolve": evolve,
                             "recursive": recursive, "seam": cfg["seam"],
                             "region": cfg.get("region"),
                             "per_cycle_wear": per_cycle_wear})
