from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import uuid
import numpy as np
from .segment import EDL, Segment
from .stages.base import Context
from . import io


def _smear_frames(smear: float, sr: int) -> int:
    return max(1, int((0.005 + smear * 0.095) * sr))


def _snap_zerocross(source: Path, frame: int, channels: str, search: int = 64) -> int:
    """Nudge `frame` to the nearest zero crossing within ±search frames."""
    start = max(0, frame - search)
    win = io.read_frames(source, start, 2 * search, channels)
    if len(win) == 0:
        return frame
    m = win.mean(axis=1)
    sign = np.signbit(m)
    crossings = np.nonzero(np.diff(sign))[0]
    if len(crossings) == 0:
        return frame
    cand = start + crossings
    return int(cand[np.argmin(np.abs(cand - frame))])


def render_edl(edl: EDL, out_path: Path, *, join: str, smear: float, channels: str) -> None:
    """Materialize the EDL to one continuous wav. cut/zerocross stream block by
    block (safe on whole-file segments); crossfade reads per grain."""
    segs = edl.segments
    sr = edl.sample_rate
    out_ch = 1 if channels in ("sum", "left") else (segs[0].channels if segs else 1)
    with io.BlockWriter(out_path, sr, out_ch) as w:
        if join == "crossfade":
            _render_crossfade(segs, w, channels, _smear_frames(smear, sr))
        else:
            for seg in segs:
                s, e = seg.start_frame, seg.end_frame
                if join == "zerocross":
                    s = _snap_zerocross(seg.source, s, channels)
                    e = max(s + 1, _snap_zerocross(seg.source, e, channels))
                for block in io.read_window(seg.source, s, e - s, channels=channels):
                    w.write(block)


def _render_crossfade(segs, w, channels, L0) -> None:
    tail = None  # previous grain's faded-out overlap, pending write
    for i, seg in enumerate(segs):
        audio = io.read_frames(seg.source, seg.start_frame, seg.n_frames, channels)
        if len(audio) == 0:
            continue
        body_start = 0
        if tail is not None:
            L = len(tail)
            head = audio[:L] * _fade(L, rising=True)[:, None]
            w.write(tail + head)
            body_start = L
        last = i == len(segs) - 1
        len_next = segs[i + 1].n_frames if not last else 0
        Ln = 0 if last else min(L0, len(audio) - body_start, len_next)
        if Ln > 0:
            w.write(audio[body_start:len(audio) - Ln])
            tail = audio[len(audio) - Ln:] * _fade(Ln, rising=False)[:, None]
        else:
            w.write(audio[body_start:])
            tail = None


def _fade(n: int, rising: bool) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False, dtype="float32")
    return np.sin(t * np.pi / 2) if rising else np.cos(t * np.pi / 2)


class Splice:
    name = "splice"

    def __init__(self, join: str = "crossfade", smear: float = 0.2):
        self.join = join
        self.smear = float(smear)

    def process(self, edl: EDL, ctx: Context) -> EDL:
        out_path = ctx.scratch_dir / f"splice_{uuid.uuid4().hex[:8]}.wav"
        render_edl(edl, out_path, join=self.join, smear=self.smear, channels=ctx.channels)
        sr, ch, n = io.info(out_path)
        rendered = Segment(source=out_path, start_frame=0, end_frame=n,
                           sample_rate=sr, channels=ch, ops=(f"splice:{self.join}",))
        edl.segments = [rendered]
        edl.record(self.name, {"join": self.join, "smear": self.smear})
        return edl
