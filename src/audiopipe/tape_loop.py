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
    render_edl(edl, path, join="cut", smear=0.0, mono=ctx.mono)
    return path


def run_tape_loop(edl: EDL, ctx: Context, cfg: dict, out_path: Path) -> None:
    """Post-chain: build `cycles` copies of the rendered loop, degrade each by
    its wear (render-once, degrade-per-cycle), concatenate with the seam join."""
    cycles = int(cfg["cycles"])
    loop_path = _loop_content(edl, ctx)

    if cycles <= 1:
        sf.write(str(out_path), io.read_frames(loop_path, 0, io.frames_of(loop_path),
                                               "independent"), io.info(loop_path)[0])
        edl.record("tape_loop", {"cycles": 1})
        return

    sr, ch, n = io.info(loop_path)
    original = io.read_frames(loop_path, 0, n, "independent")
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
    render_edl(seam_edl, out_path, join=cfg["seam"], smear=0.2, mono="independent")
    edl.record("tape_loop", {"cycles": cycles, "evolve": evolve,
                             "recursive": recursive, "seam": cfg["seam"],
                             "per_cycle_wear": per_cycle_wear})
