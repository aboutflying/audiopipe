from __future__ import annotations
from dataclasses import replace
import uuid
from .segment import EDL, Segment
from .stages.base import Context
from .mapping import grain_frames


def _cut_segment(seg: Segment, strategy: str, amount: float, jitter: float,
                 rng) -> list[Segment]:
    base = grain_frames(amount, seg.sample_rate)
    bounds = [seg.start_frame]
    pos = seg.start_frame
    while pos < seg.end_frame:
        step = base if strategy == "grid" else rng.randint(base // 2 or 1, base * 3 // 2 or 1)
        nxt = pos + step
        if jitter > 0 and nxt < seg.end_frame:
            nxt += int(rng.uniform(-jitter, jitter) * base)
        nxt = min(max(nxt, pos + 1), seg.end_frame)
        bounds.append(nxt)
        pos = nxt
    if bounds[-1] != seg.end_frame:
        bounds[-1] = seg.end_frame
    return _from_bounds(seg, bounds, strategy)


def _from_bounds(seg: Segment, bounds: list[int], strategy: str) -> list[Segment]:
    out = []
    for s, e in zip(bounds, bounds[1:]):
        if e > s:
            out.append(replace(seg, start_frame=s, end_frame=e,
                               ops=seg.ops + (f"slice:{strategy}",),
                               seg_id=uuid.uuid4().hex[:8]))
    return out


def _cut_onset(seg: Segment, strategy: str) -> list[Segment]:
    """Cut at detected onsets (strategy=onset) or non-silent spans (silence)."""
    from . import analyze
    if strategy == "onset":
        offs = analyze.onset_offsets(seg)
        bounds = [seg.start_frame] + [seg.start_frame + o for o in offs] + [seg.end_frame]
        bounds = sorted(set(bounds))
        return _from_bounds(seg, bounds, strategy)
    spans = analyze.silence_spans(seg)
    return [replace(seg, start_frame=seg.start_frame + s, end_frame=seg.start_frame + e,
                    ops=seg.ops + ("slice:silence",), seg_id=uuid.uuid4().hex[:8])
            for s, e in spans]


class Segmenter:
    name = "slice"

    def __init__(self, strategy: str = "grid", amount: float = 0.6, jitter: float = 0.3):
        self.strategy = strategy
        self.amount = float(amount)
        self.jitter = float(jitter)

    def process(self, edl: EDL, ctx: Context) -> EDL:
        if self.strategy not in ("grid", "random", "onset", "silence"):
            raise ValueError(f"unknown slice strategy {self.strategy!r}")
        out: list[Segment] = []
        for seg in edl.segments:
            if self.strategy in ("onset", "silence"):
                out.extend(_cut_onset(seg, self.strategy))
            else:
                out.extend(_cut_segment(seg, self.strategy, self.amount, self.jitter, ctx.rng))
        edl.segments = out
        edl.record(self.name, {"strategy": self.strategy, "amount": self.amount,
                               "jitter": self.jitter})
        return edl
