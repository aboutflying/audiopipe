from __future__ import annotations
from dataclasses import replace
import uuid
from .segment import EDL, Segment
from .stages.base import Context
from .mapping import grain_frames


def _cut_segment(seg: Segment, mode: str, density: float, drift: float,
                 rng) -> list[Segment]:
    base = grain_frames(density, seg.sample_rate)
    bounds = [seg.start_frame]
    pos = seg.start_frame
    while pos < seg.end_frame:
        step = base if mode == "grid" else rng.randint(base // 2 or 1, base * 3 // 2 or 1)
        nxt = pos + step
        if drift > 0 and nxt < seg.end_frame:
            nxt += int(rng.uniform(-drift, drift) * base)
        nxt = min(max(nxt, pos + 1), seg.end_frame)
        bounds.append(nxt)
        pos = nxt
    if bounds[-1] != seg.end_frame:
        bounds[-1] = seg.end_frame
    return _from_bounds(seg, bounds, mode)


def _from_bounds(seg: Segment, bounds: list[int], mode: str) -> list[Segment]:
    out = []
    for s, e in zip(bounds, bounds[1:]):
        if e > s:
            out.append(replace(seg, start_frame=s, end_frame=e,
                               ops=seg.ops + (f"grain:{mode}",),
                               seg_id=uuid.uuid4().hex[:8]))
    return out


def _cut_onset(seg: Segment, mode: str) -> list[Segment]:
    """Cut at detected onsets (mode=onset) or non-silent spans (silence)."""
    from . import analyze
    if mode == "onset":
        offs = analyze.onset_offsets(seg)
        bounds = [seg.start_frame] + [seg.start_frame + o for o in offs] + [seg.end_frame]
        bounds = sorted(set(bounds))
        return _from_bounds(seg, bounds, mode)
    spans = analyze.silence_spans(seg)
    return [replace(seg, start_frame=seg.start_frame + s, end_frame=seg.start_frame + e,
                    ops=seg.ops + ("grain:silence",), seg_id=uuid.uuid4().hex[:8])
            for s, e in spans]


class Segmenter:
    name = "grain"

    def __init__(self, mode: str = "grid", density: float = 0.6, drift: float = 0.3):
        self.mode = mode
        self.density = float(density)
        self.drift = float(drift)

    def process(self, edl: EDL, ctx: Context) -> EDL:
        if self.mode not in ("grid", "random", "onset", "silence"):
            raise ValueError(f"unknown grain mode {self.mode!r}")
        out: list[Segment] = []
        for seg in edl.segments:
            if self.mode in ("onset", "silence"):
                out.extend(_cut_onset(seg, self.mode))
            else:
                out.extend(_cut_segment(seg, self.mode, self.density, self.drift, ctx.rng))
        edl.segments = out
        edl.record(self.name, {"mode": self.mode, "density": self.density,
                               "drift": self.drift})
        return edl
