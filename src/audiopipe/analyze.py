from __future__ import annotations
from pathlib import Path
import numpy as np
import librosa

from .segment import Segment
from . import io


def _mono(seg: Segment) -> np.ndarray:
    """Read a segment's audio summed to mono (1D float32)."""
    a = io.read_frames(seg.source, seg.start_frame, seg.n_frames, channels="sum")
    return a[:, 0] if a.ndim == 2 else a


def onset_offsets(seg: Segment) -> list[int]:
    """Frame offsets (relative to seg.start_frame) of detected onsets."""
    y = _mono(seg)
    if len(y) == 0:
        return []
    frames = librosa.onset.onset_detect(y=y, sr=seg.sample_rate, units="samples",
                                        backtrack=True)
    return [int(f) for f in frames if 0 < f < len(y)]


def silence_spans(seg: Segment, top_db: float = 30.0) -> list[tuple[int, int]]:
    """Non-silent [start, end) offsets within the segment."""
    y = _mono(seg)
    if len(y) == 0:
        return []
    intervals = librosa.effects.split(y, top_db=top_db)
    return [(int(s), int(e)) for s, e in intervals]


def feature(seg: Segment, key: str) -> float:
    """Scalar feature used by feel: sort. brightness=spectral centroid,
    loudness=RMS, duration=length in frames."""
    if key == "duration":
        return float(seg.n_frames)
    y = _mono(seg)
    if len(y) == 0:
        return 0.0
    if key == "loudness":
        return float(np.sqrt(np.mean(y ** 2)))
    if key == "brightness":
        return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=seg.sample_rate)))
    raise ValueError(f"unknown feature {key!r}")
