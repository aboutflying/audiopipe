from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest


def write_tone(path: Path, seconds: float = 2.0, sr: int = 16000, freq: float = 220.0,
               channels: int = 1) -> Path:
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * freq * t).astype("float32")
    data = sig if channels == 1 else np.stack([sig] * channels, axis=1)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


@pytest.fixture
def tone(tmp_path):
    return write_tone(tmp_path / "tone.wav")
