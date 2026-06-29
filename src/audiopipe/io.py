from __future__ import annotations
from pathlib import Path
from typing import Iterator
import numpy as np
import soundfile as sf


def info(path: Path) -> tuple[int, int, int]:
    """(sample_rate, channels, n_frames). No audio loaded."""
    i = sf.info(str(path))
    return i.samplerate, i.channels, i.frames


def frames_of(path: Path) -> int:
    return sf.info(str(path)).frames


def _apply_mono(block: np.ndarray, mono: str) -> np.ndarray:
    """block is (n, channels). Return (n, channels') per policy."""
    if block.ndim == 1:
        block = block[:, None]
    if block.shape[1] == 1 or mono == "independent":
        return block
    if mono == "sum":
        return block.mean(axis=1, keepdims=True)
    if mono == "left":
        return block[:, :1]
    raise ValueError(f"unknown mono policy: {mono}")


def read_window(path: Path, start: int, n: int, mono: str = "independent",
                block: int = 1 << 16) -> Iterator[np.ndarray]:
    """Yield (frames, channels) float32 blocks covering [start, start+n).
    Windowed: never loads the whole file."""
    with sf.SoundFile(str(path)) as f:
        f.seek(start)
        remaining = n
        while remaining > 0:
            chunk = f.read(min(block, remaining), dtype="float32", always_2d=True)
            if len(chunk) == 0:
                break
            remaining -= len(chunk)
            yield _apply_mono(chunk, mono)


def read_frames(path: Path, start: int, n: int, mono: str = "independent") -> np.ndarray:
    """Read [start, start+n) fully as one (frames, channels) array. For small
    grains and boundary windows only — use read_window for long spans."""
    with sf.SoundFile(str(path)) as f:
        f.seek(start)
        block = f.read(n, dtype="float32", always_2d=True)
    return _apply_mono(block, mono)


class BlockWriter:
    """Block-write output rather than accumulating one giant array."""
    def __init__(self, path: Path, sample_rate: int, channels: int):
        self._f = sf.SoundFile(str(path), mode="w", samplerate=sample_rate,
                               channels=channels, subtype="PCM_24")

    def write(self, block: np.ndarray) -> None:
        self._f.write(block)

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def resample(block: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample (frames, channels) audio. Stub seam; not exercised in M1/M2.
    Linear interpolation keeps it dependency-free until a stage needs better.
    ponytail: linear resample; swap for scipy/librosa if quality matters (M3+)."""
    if src_sr == dst_sr:
        return block
    if block.ndim == 1:
        block = block[:, None]
    n_out = round(block.shape[0] * dst_sr / src_sr)
    xp = np.arange(block.shape[0])
    x = np.linspace(0, block.shape[0] - 1, n_out)
    return np.stack([np.interp(x, xp, block[:, c]) for c in range(block.shape[1])],
                    axis=1).astype("float32")
