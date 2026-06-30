from __future__ import annotations
from dataclasses import replace
import uuid
import numpy as np
from .segment import EDL, Segment
from .stages.base import Context
from . import io


class Warp:
    """Tape-style per-segment transforms: play a grain backwards and/or change
    its playback rate (varispeed — pitch follows speed). Materializes changed
    segments to scratch; unchanged ones stay reference-only."""
    name = "vari"

    def __init__(self, reverse: float = 0.0, speed: float = 1.0,
                 wobble: float = 0.0):
        self.reverse = float(reverse)          # 0..1 probability a grain is reversed
        self.speed = float(speed)              # rate multiplier (>1 faster+higher)
        self.wobble = float(wobble)            # per-grain random speed spread

    def process(self, edl: EDL, ctx: Context) -> EDL:
        out: list[Segment] = []
        for seg in edl.segments:
            r = self._render(seg, ctx)
            if r is not None:
                out.append(r)
        edl.segments = out
        edl.record(self.name, {"reverse": self.reverse, "speed": self.speed,
                               "wobble": self.wobble})
        return edl

    def _render(self, seg: Segment, ctx: Context) -> Segment | None:
        # draw both randoms up front so the RNG stream is stable per segment
        do_reverse = ctx.rng.random() < self.reverse
        speed = self.speed * (1 + ctx.rng.uniform(-self.wobble, self.wobble))
        speed = max(0.1, speed)
        if not do_reverse and abs(speed - 1.0) < 1e-6:
            return seg                          # untouched: stay reference-only

        audio = io.read_frames(seg.source, seg.start_frame, seg.n_frames, ctx.channels)
        if len(audio) == 0:
            return None
        ops = []
        if do_reverse:
            audio = np.ascontiguousarray(audio[::-1])
            ops.append("rev")
        if abs(speed - 1.0) >= 1e-6:
            audio = io.resample_to(audio, max(1, round(len(audio) / speed)))
            ops.append(f"speed{speed:.2f}")

        path = ctx.scratch_dir / f"vari_{uuid.uuid4().hex[:8]}.wav"
        with io.BlockWriter(path, seg.sample_rate, audio.shape[1]) as w:
            w.write(audio)
        return replace(seg, source=path, start_frame=0, end_frame=len(audio),
                       ops=seg.ops + tuple(ops), seg_id=uuid.uuid4().hex[:8])
