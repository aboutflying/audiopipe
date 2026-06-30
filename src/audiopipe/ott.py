from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import uuid
import numpy as np
from .segment import EDL, Segment
from .stages.base import Context
from . import io

# 3-band crossovers (Hz). Complementary split (mid = full - low - high) so the
# bands sum back to the input exactly when uncompressed -> transparent at depth 0.
_F1, _F2 = 250.0, 2500.0


def _sos(sr, cutoff, btype):
    from scipy.signal import butter
    return butter(4, cutoff, btype=btype, fs=sr, output="sos")


def _filt(sos, x):
    from scipy.signal import sosfilt
    return sosfilt(sos, x, axis=0).astype("float32")


def _envelope(mono, sr, tau=0.012):
    """One-pole smoothing of the rectified signal (stereo-linked detector)."""
    from scipy.signal import lfilter
    a = float(np.exp(-1.0 / (tau * sr)))
    return lfilter([1 - a], [1.0, -a], np.abs(mono)).astype("float32")


def _compress_band(band, sr, depth):
    """Upward + downward compression around a depth-scaled threshold. This is the
    OTT signature: loud is squashed AND quiet is lifted toward the threshold."""
    env = _envelope(band.mean(axis=1), sr) + 1e-9
    env_db = 20 * np.log10(env)
    thresh_db = -6 - 24 * depth          # lower threshold = more compression
    ratio_down = 1 + 7 * depth
    ratio_up = 1 + 3 * depth
    over = env_db - thresh_db
    gain_db = np.zeros_like(env_db)
    down = over > 0
    gain_db[down] = -over[down] * (1 - 1 / ratio_down)          # squash above
    up = (~down) & (env_db > -60.0)                            # lift below (not pure silence)
    gain_db[up] = -over[up] * (1 - 1 / ratio_up)
    gain = 10 ** (np.clip(gain_db, -24, 24) / 20)
    return band * gain[:, None]


def ott_process(audio: np.ndarray, sr: int, depth: float) -> np.ndarray:
    """Multiband upward+downward compression (OTT-style). depth 0..1; 0 = bypass."""
    if depth <= 0:
        return audio
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = audio.astype("float32")
    low = _filt(_sos(sr, _F1, "low"), audio)
    high = _filt(_sos(sr, _F2, "high"), audio)
    mid = audio - low - high
    out = (_compress_band(low, sr, depth) + _compress_band(mid, sr, depth)
           + _compress_band(high, sr, depth))
    out *= 10 ** ((18 * depth) / 20)     # makeup: strong, to stay loud as it slams
    return np.tanh(out).astype("float32")  # soft-clip the slam (no harsh digital clip)


def ott_file(path: Path, depth: float) -> None:
    """Apply OTT to a rendered file in place (the whole-output / master position)."""
    import soundfile as sf
    sr, ch, n = io.info(path)
    out = ott_process(io.read_frames(path, 0, n, "keep"), sr, depth)
    sf.write(str(path), out, sr)


class Ott:
    """Extreme multiband compressor. where='grain' slams each grain in the chain;
    where='output' runs as a master pass on the final render (via the runner)."""
    name = "ott"

    def __init__(self, depth: float = 0.0, where: str = "grain"):
        self.depth = float(depth)
        self.where = where

    def process(self, edl: EDL, ctx: Context) -> EDL:
        if self.where not in ("grain", "output"):
            raise ValueError(f"ott.where must be 'grain' or 'output', got {self.where!r}")
        # as a chain stage it only acts per-grain; 'output' is handled by the runner
        if self.where == "grain" and self.depth > 0:
            edl.segments = [s for s in (self._render(seg, ctx) for seg in edl.segments)
                            if s is not None]
        edl.record(self.name, {"depth": self.depth, "where": self.where})
        return edl

    def _render(self, seg: Segment, ctx: Context) -> Segment | None:
        audio = io.materialize(seg, ctx.channels)
        if len(audio) == 0:
            return None
        out = ott_process(audio, seg.sample_rate, self.depth)
        return replace(seg, start_frame=0, end_frame=len(out),
                       ops=seg.ops + ("ott",), seg_id=uuid.uuid4().hex[:8], audio=out)
