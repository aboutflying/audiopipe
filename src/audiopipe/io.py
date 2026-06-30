from __future__ import annotations
from pathlib import Path
from typing import Iterator
import shutil
import subprocess
import numpy as np
import soundfile as sf


def ensure_readable(path: Path, scratch_dir: Path) -> Path:
    """Return a libsndfile-readable path for `path`. WAV/FLAC/etc. pass through;
    formats libsndfile can't decode (M4A/AAC/ALAC) are transcoded to WAV via
    macOS `afconvert`. Raises if unreadable and no transcoder is available."""
    path = Path(path)
    try:
        sf.info(str(path))
        return path
    except sf.LibsndfileError:
        pass  # not a libsndfile format; try CoreAudio
    afconvert = shutil.which("afconvert")
    if not afconvert:
        raise RuntimeError(f"{path.name}: not a libsndfile format and `afconvert` "
                           "(macOS) is unavailable to transcode it")
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    out = scratch_dir / f"{path.stem}.decoded.wav"
    subprocess.run([afconvert, "-f", "WAVE", "-d", "LEI16", str(path), str(out)],
                   check=True, capture_output=True)
    return out


def info(path: Path) -> tuple[int, int, int]:
    """(sample_rate, channels, n_frames). No audio loaded."""
    i = sf.info(str(path))
    return i.samplerate, i.channels, i.frames


def frames_of(path: Path) -> int:
    return sf.info(str(path)).frames


def _apply_channels(block: np.ndarray, channels: str) -> np.ndarray:
    """block is (n, channels). Return (n, channels') per the source.channels policy."""
    if block.ndim == 1:
        block = block[:, None]
    if block.shape[1] == 1 or channels == "keep":
        return block
    if channels == "sum":
        return block.mean(axis=1, keepdims=True)
    if channels == "left":
        return block[:, :1]
    raise ValueError(f"unknown channel policy: {channels}")


def read_window(path: Path, start: int, n: int, channels: str = "keep",
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
            yield _apply_channels(chunk, channels)


def read_frames(path: Path, start: int, n: int, channels: str = "keep") -> np.ndarray:
    """Read [start, start+n) fully as one (frames, channels) array. For small
    grains and boundary windows only — use read_window for long spans."""
    with sf.SoundFile(str(path)) as f:
        f.seek(start)
        block = f.read(n, dtype="float32", always_2d=True)
    return _apply_channels(block, channels)


def materialize(seg, channels: str) -> np.ndarray:
    """A segment's audio as a (frames, channels) array: the in-memory buffer if a
    stage already rendered one, otherwise a windowed read from its source file."""
    if seg.audio is not None:
        return seg.audio
    return read_frames(seg.source, seg.start_frame, seg.n_frames, channels)


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


def resample_to(block: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly resample (frames, channels) audio to n_out frames. Varispeed:
    fewer frames played at the same rate => faster + higher pitch (tape style).
    ponytail: linear interp; swap for scipy/librosa if quality matters."""
    if block.ndim == 1:
        block = block[:, None]
    n_in = block.shape[0]
    if n_out == n_in or n_in == 0:
        return block.astype("float32")
    xp = np.arange(n_in)
    x = np.linspace(0, n_in - 1, n_out)
    return np.stack([np.interp(x, xp, block[:, c]) for c in range(block.shape[1])],
                    axis=1).astype("float32")


def resample(block: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample to a new sample rate (preserving pitch/duration). Stub seam,
    not exercised in M1/M2."""
    if src_sr == dst_sr:
        return block
    n = block.shape[0] if block.ndim > 1 else len(block)
    return resample_to(block, round(n * dst_sr / src_sr))
