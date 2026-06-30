from __future__ import annotations
from dataclasses import replace
import uuid
from .segment import EDL, Segment
from .stages.base import Context
from .mapping import fx_params
from . import io


def _build_board(params: dict):
    """Construct a Pedalboard from concrete params. Imported lazily so pedalboard
    stays an optional M4 dependency, not required to run M1-M3 chains."""
    import pedalboard as pb
    fx = []
    if "drive_db" in params:
        fx.append(pb.Distortion(drive_db=params["drive_db"]))
    if "cutoff_hz" in params:
        fx.append(pb.LowpassFilter(cutoff_frequency_hz=params["cutoff_hz"]))
    if "chorus_mix" in params:
        fx.append(pb.Chorus(mix=params["chorus_mix"]))
    if "reverb_room" in params:
        fx.append(pb.Reverb(room_size=params["reverb_room"],
                            wet_level=params["reverb_wet"]))
    return pb.Pedalboard(fx)


class Dsp:
    """Sample-transforming stage: applies a pedalboard effect chain to each
    segment, writing rendered audio to scratch (segments become scratch-backed)."""
    name = "fx"

    def __init__(self, drive: float = 0.2, tone: float = 0.3,
                 chorus: float = 0.0, reverb: float = 0.25):
        self.dials = {"drive": float(drive), "tone": float(tone),
                      "chorus": float(chorus), "reverb": float(reverb)}

    def process(self, edl: EDL, ctx: Context) -> EDL:
        params = fx_params(self.dials)
        if not params:
            edl.record(self.name, {**self.dials, "effects": []})
            return edl
        board = _build_board(params)
        out: list[Segment] = []
        for seg in edl.segments:
            rendered = self._render(seg, board, ctx)
            if rendered is not None:
                out.append(rendered)
        edl.segments = out
        edl.record(self.name, {**self.dials, "effects": sorted(params)})
        return edl

    def _render(self, seg: Segment, board, ctx: Context) -> Segment | None:
        path = ctx.scratch_dir / f"fx_{uuid.uuid4().hex[:8]}.wav"
        writer = None
        total = 0
        first = True
        for block in io.read_window(seg.source, seg.start_frame, seg.n_frames, channels=ctx.channels):
            y = board(block, seg.sample_rate, reset=first)  # reset state per grain
            first = False
            if writer is None:
                writer = io.BlockWriter(path, seg.sample_rate, y.shape[1])
            writer.write(y)
            total += len(y)
        if writer is None:
            return None
        writer.close()
        return replace(seg, source=path, start_frame=0, end_frame=total,
                       ops=seg.ops + ("fx",), seg_id=uuid.uuid4().hex[:8])
