from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt


def _lowpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    cutoff = max(200.0, min(cutoff, sr / 2 - 1))
    sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
    return sosfilt(sos, audio, axis=0).astype("float32")


def degrade(audio: np.ndarray, sr: int, wear: float, rng) -> np.ndarray:
    """Apply tape wear scaled by `wear` in [0,1]. wear 0 returns audio unchanged.
    Lowpass roll-off + gain attenuation + random dropouts, all from `rng` so the
    whole loop reproduces from the master seed."""
    if wear <= 0:
        return audio.astype("float32", copy=True)
    if audio.ndim == 1:
        audio = audio[:, None]
    n = len(audio)

    # progressive dulling: full bandwidth at wear 0 -> ~1.5kHz at wear 1
    cutoff = sr / 2 * (1 - wear) + 1500 * wear
    out = _lowpass(audio, sr, cutoff)

    # level loss up to -6 dB-ish
    out *= (1 - 0.5 * wear)

    # random dropouts: more, longer holes as wear climbs
    n_drops = int(wear * 8)
    hole = int(0.02 * sr)
    for _ in range(n_drops):
        start = rng.randint(0, max(0, n - hole))
        out[start:start + hole] = 0.0
    return out
