from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import uuid
import numpy as np
from .segment import EDL, Segment
from .stages.base import Context
from .mapping import fx_params
from . import io


def fx_file(path: Path, dials: dict) -> None:
    """Apply the fx board to a rendered file in place (the master/glue position).
    Reverb/chorus get ~3 s of appended silence first so the tail rings out past
    the end instead of truncating — the reason glue reverb lives here and not
    per grain (where every tail would be cut at the grain boundary)."""
    import soundfile as sf
    params = fx_params(dials)
    if not params:
        return
    sr, ch, n = io.info(path)
    audio = io.read_frames(path, 0, n, "keep")
    if "reverb_room" in params or "chorus_mix" in params:
        audio = np.concatenate([audio, np.zeros((3 * sr, audio.shape[1]), dtype="float32")])
    out = _build_board(params, sr)(audio, sr, reset=True)
    sf.write(str(path), out, sr)


def _build_board(params: dict, sr: int):
    """Construct a Pedalboard from concrete params. Imported lazily so pedalboard
    stays an optional M4 dependency, not required to run M1-M3 chains."""
    import pedalboard as pb
    fx = []
    if "drive_db" in params:
        fx.append(pb.Distortion(drive_db=params["drive_db"]))
    if "cutoff_hz" in params:
        cutoff = min(params["cutoff_hz"], sr / 2 * 0.95)   # keep below Nyquist (filter blows up otherwise)
        fx.append(pb.LowpassFilter(cutoff_frequency_hz=cutoff))
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
        board = _build_board(params, edl.sample_rate)
        out: list[Segment] = []
        for seg in edl.segments:
            rendered = self._render(seg, board, ctx)
            if rendered is not None:
                out.append(rendered)
        edl.segments = out
        edl.record(self.name, {**self.dials, "effects": sorted(params)})
        return edl

    def _render(self, seg: Segment, board, ctx: Context) -> Segment | None:
        audio = io.materialize(seg, ctx.channels)
        if len(audio) == 0:
            return None
        out = board(audio, seg.sample_rate, reset=True)
        return replace(seg, start_frame=0, end_frame=len(out),
                       ops=seg.ops + ("fx",), seg_id=uuid.uuid4().hex[:8], audio=out)
