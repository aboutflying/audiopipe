from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt


def _lowpass(audio: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    cutoff = max(200.0, min(cutoff, sr / 2 - 1))
    sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
    return sosfilt(sos, audio, axis=0).astype("float32")


def degrade(audio: np.ndarray, sr: int, wear: float, rng=None) -> np.ndarray:
    """Tape wear scaled by `wear` in [0,1]: high-frequency roll-off + level loss.
    wear 0 returns audio unchanged. This is the per-cycle disintegration; hiss,
    dropouts, and flutter are separate dials applied on top."""
    if wear <= 0:
        return audio.astype("float32", copy=True)
    if audio.ndim == 1:
        audio = audio[:, None]
    # progressive dulling on a perceptual (log-frequency) curve so even moderate
    # wear is audible regardless of sample rate: ~18 kHz at wear 0 -> 250 Hz at 1.
    cutoff = min(sr / 2 * 0.95, 18000.0 * (250.0 / 18000.0) ** wear)
    out = _lowpass(audio, sr, cutoff)
    out *= (1 - 0.5 * wear)          # level loss up to ~-6 dB
    return out


def add_hiss(audio: np.ndarray, level: float, rng) -> np.ndarray:
    """Additive tape noise floor. level 0..1 (level 1 ~= -34 dB). Seeded from
    `rng` so the hiss reproduces from the master seed."""
    if level <= 0:
        return audio
    if audio.ndim == 1:
        audio = audio[:, None]
    np_rng = np.random.default_rng(rng.randint(0, 2 ** 32 - 1))
    noise = np_rng.standard_normal(audio.shape).astype("float32") * (0.02 * level)
    return (audio + noise).astype("float32")


def add_flutter(audio: np.ndarray, sr: int, amount: float, rng) -> np.ndarray:
    """Wow & flutter on the whole buffer: a slow (wow) + fast (flutter) LFO warps
    the playback timebase, so pitch drifts. amount 0..1 -> up to ~3% deviation.
    Output length is unchanged. Phases drawn from `rng` for reproducibility."""
    if amount <= 0:
        return audio
    if audio.ndim == 1:
        audio = audio[:, None]
    n = len(audio)
    t = np.arange(n) / sr
    ph_wow, ph_flutter = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    lfo = 0.7 * np.sin(2 * np.pi * 0.6 * t + ph_wow) + 0.3 * np.sin(2 * np.pi * 6.0 * t + ph_flutter)
    speed = 1 + (0.03 * amount) * lfo
    pos = np.cumsum(speed)
    pos = pos / pos[-1] * (n - 1)       # warped sample positions, normalized to span
    xp = np.arange(n)
    return np.stack([np.interp(pos, xp, audio[:, c]) for c in range(audio.shape[1])],
                    axis=1).astype("float32")
