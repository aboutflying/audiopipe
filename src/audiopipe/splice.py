from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import uuid
import numpy as np
import soundfile as sf
from .segment import EDL, Segment
from .stages.base import Context
from . import io


def _fade_frames(fade: float, sr: int) -> int:
    return max(1, int((0.005 + fade * 0.095) * sr))


def _print_dropouts(path: Path, amount: float, rng) -> None:
    """Bake random short dropouts into the rendered file *in place* (tape oxide
    shedding 'printed' at render time, so a tape_loop repeats the same holes).
    amount 0..1 scales the count; each hole is ~10-50 ms. Each hole gets a short
    (~3 ms) fade out/in at its edges so it doesn't click. Drawn from `rng`."""
    sr, ch, n = io.info(path)
    holes = []
    for _ in range(int(amount * 12)):
        hole = int(rng.uniform(0.01, 0.05) * sr)
        holes.append((rng.randint(0, max(0, n - hole)), hole))
    if not holes:
        return
    fade = max(1, int(0.003 * sr))
    with sf.SoundFile(str(path), mode="r+") as f:
        for start, hole in holes:
            f_len = min(fade, hole // 2)
            if f_len > 0:
                ramp = np.linspace(1.0, 0.0, f_len, dtype="float32")[:, None]
                f.seek(start); head = f.read(f_len, dtype="float32", always_2d=True)
                f.seek(start); f.write(head * ramp)                          # fade out
                f.seek(start + hole - f_len); tail = f.read(f_len, dtype="float32", always_2d=True)
                f.seek(start + hole - f_len); f.write(tail * ramp[::-1])     # fade in
            mid = hole - 2 * f_len
            if mid > 0:
                f.seek(start + f_len)
                f.write(np.zeros((mid, ch), dtype="float32"))


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


def _zerocross_trim(a: np.ndarray, search: int = 64) -> np.ndarray:
    """Trim an in-memory grain's edges to the nearest zero crossings."""
    m = a.mean(axis=1)
    s, e = 0, len(a)
    head = np.nonzero(np.diff(np.signbit(m[:search + 1])))[0]
    if len(head):
        s = int(head[0]) + 1
    base = max(0, len(a) - search - 1)
    tail = np.nonzero(np.diff(np.signbit(m[base:])))[0]
    if len(tail):
        e = base + int(tail[-1]) + 1
    return a[s:e] if e > s else a


def render_edl(edl: EDL, out_path: Path, *, join: str, fade: float, channels: str) -> None:
    """Materialize the EDL to one continuous wav. Grains with an in-memory buffer
    are written directly; reference-only grains stream block by block from source
    (keeps long, un-effected inputs windowed). crossfade overlaps per grain."""
    segs = edl.segments
    sr = edl.sample_rate
    out_ch = 1 if channels in ("sum", "left") else (segs[0].channels if segs else 1)
    with io.BlockWriter(out_path, sr, out_ch) as w:
        if join == "crossfade":
            _render_crossfade(segs, w, channels, _fade_frames(fade, sr))
        else:
            for seg in segs:
                if seg.audio is not None:
                    w.write(_zerocross_trim(seg.audio) if join == "zerocross" else seg.audio)
                    continue
                s, e = seg.start_frame, seg.end_frame
                if join == "zerocross":
                    s = _snap_zerocross(seg.source, s, channels)
                    e = max(s + 1, _snap_zerocross(seg.source, e, channels))
                for block in io.read_window(seg.source, s, e - s, channels=channels):
                    w.write(block)


def _render_crossfade(segs, w, channels, L0) -> None:
    tail = None  # previous grain's faded-out overlap, pending write
    for i, seg in enumerate(segs):
        audio = io.materialize(seg, channels)
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

    def __init__(self, join: str = "cut", fade: float = 0.0, dropouts: float = 0.0):
        self.join = join
        self.fade = float(fade)
        self.dropouts = float(dropouts)

    def process(self, edl: EDL, ctx: Context) -> EDL:
        out_path = ctx.scratch_dir / f"splice_{uuid.uuid4().hex[:8]}.wav"
        render_edl(edl, out_path, join=self.join, fade=self.fade, channels=ctx.channels)
        if self.dropouts > 0:
            _print_dropouts(out_path, self.dropouts, ctx.rng)
        sr, ch, n = io.info(out_path)
        ops = (f"splice:{self.join}",) + (("dropouts",) if self.dropouts > 0 else ())
        rendered = Segment(source=out_path, start_frame=0, end_frame=n,
                           sample_rate=sr, channels=ch, ops=ops)
        edl.segments = [rendered]
        edl.record(self.name, {"join": self.join, "fade": self.fade,
                               "dropouts": self.dropouts})
        return edl
