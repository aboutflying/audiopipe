from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import yaml
import pytest

pytest.importorskip("scipy")

from audiopipe.segment import EDL
from audiopipe.stages.base import Context
from audiopipe.ott import Ott, ott_process
from audiopipe.segmenter import Segmenter
from audiopipe import io
from audiopipe.runner import render_one
from .conftest import write_tone


def _ctx(tmp_path, seed=42, channels="keep"):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    return Context(scratch_dir=Path(tmp_path), rng=random.Random(seed), channels=channels)


def _loud_quiet(sr=16000):
    t = np.linspace(0, 1, sr, endpoint=False)
    loud = 0.8 * np.sin(2 * np.pi * 440 * t)
    quiet = 0.04 * np.sin(2 * np.pi * 440 * t)
    return np.concatenate([loud, quiet]).astype("float32")[:, None]


def test_ott_depth0_is_bypass():
    x = _loud_quiet()
    assert np.array_equal(ott_process(x, 16000, 0.0), x)


def test_ott_reduces_crest_factor():
    # upward+downward compression shrinks the gap between loud and quiet sections
    x = _loud_quiet()
    n = len(x) // 2
    rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
    before = rms(x[:n]) / rms(x[n:])
    y = ott_process(x, 16000, 0.8)
    after = rms(y[:n]) / rms(y[n:])
    assert after < before                     # dynamic range compressed
    assert np.max(np.abs(y)) <= 1.0           # bounded


def test_ott_deterministic():
    x = _loud_quiet()
    assert np.array_equal(ott_process(x, 16000, 0.7), ott_process(x, 16000, 0.7))


def test_ott_per_grain_stage(tmp_path, tone):
    edl = EDL.single(Path(tone), io.frames_of(tone), 16000, 1, seed=1)
    e = Segmenter("grid", 0.6, 0.0).process(edl, _ctx(tmp_path))
    out = Ott(depth=0.7, where="grain").process(e, _ctx(tmp_path))
    assert all("ott" in s.ops for s in out.segments)          # every grain slammed
    assert all(s.audio is not None for s in out.segments)     # rendered in-memory


def test_ott_output_where_is_noop_in_chain(tmp_path, tone):
    # where='output' must NOT process in the chain (the runner handles it)
    edl = EDL.single(Path(tone), io.frames_of(tone), 16000, 1, seed=1)
    e = Segmenter("grid", 0.6, 0.0).process(edl, _ctx(tmp_path))
    srcs = [s.source for s in e.segments]
    out = Ott(depth=0.9, where="output").process(e, _ctx(tmp_path))
    assert [s.source for s in out.segments] == srcs           # untouched


def test_ott_whole_output_via_cli(tmp_path):
    tone = write_tone(tmp_path / "in.wav", seconds=3.0)
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({
        "chain": ["grain", "splice"],
        "ott": {"depth": 0.8, "where": "output"}}))
    out = tmp_path / "out.wav"
    _, edl, _ = render_one(tone, cfg, out, tmp_path / "scratch")
    assert out.exists() and io.frames_of(out) > 0
    # the per-grain ott stage isn't in the chain; history records it didn't run there
    assert not any(h["stage"] == "ott" for h in edl.history)
