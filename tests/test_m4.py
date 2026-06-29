from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import yaml
import pytest

pytest.importorskip("pedalboard")

from audiopipe.segment import EDL
from audiopipe.stages.base import Context
from audiopipe.dsp import Dsp
from audiopipe.mapping import dsp_params
from audiopipe import io
from audiopipe.runner import render_one
from .conftest import write_tone


def _ctx(tmp_path, seed=42, channels="sum"):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    return Context(scratch_dir=Path(tmp_path), rng=random.Random(seed), channels=channels)


def _single(tone, sr=16000):
    _, ch, n = io.info(tone)
    return EDL.single(Path(tone), n, sr, ch, seed=42)


def test_dsp_params_off_when_zero():
    assert dsp_params({"drive": 0, "filter": 0, "chorus": 0, "reverb": 0}) == {}
    p = dsp_params({"drive": 1.0, "filter": 0, "chorus": 0, "reverb": 0})
    assert p == {"drive_db": 30.0}


def test_dsp_all_zero_is_passthrough(tmp_path, tone):
    edl = _single(tone)
    src = edl.segments[0].source
    out = Dsp(drive=0, filter=0, chorus=0, reverb=0).process(edl, _ctx(tmp_path))
    assert out.segments[0].source == src          # untouched, no scratch render
    assert out.history[-1]["stage"] == "dsp"


def test_dsp_renders_to_scratch_and_transforms(tmp_path, tone):
    edl = _single(tone)
    n_in = edl.segments[0].n_frames
    out = Dsp(drive=0.5, filter=0.6, chorus=0.0, reverb=0.0).process(edl, _ctx(tmp_path))
    seg = out.segments[0]
    assert tmp_path in Path(seg.source).parents          # scratch-backed
    assert "dsp" in seg.ops
    assert seg.start_frame == 0 and seg.n_frames == n_in  # filter/drive preserve length
    dry, _ = sf.read(str(tone), always_2d=True)
    wet, _ = sf.read(str(seg.source), always_2d=True)
    assert not np.allclose(dry[:, :1].mean(1, keepdims=True), wet, atol=1e-3)


def test_filter_reduces_brightness(tmp_path, tone):
    # a dark lowpass should cut high-frequency energy vs the dry signal
    edl = _single(tone)
    out = Dsp(drive=0, filter=0.95, chorus=0, reverb=0).process(edl, _ctx(tmp_path))
    dry, _ = sf.read(str(tone))
    wet, _ = sf.read(str(out.segments[0].source))
    hf_dry = float(np.sum(np.diff(dry) ** 2))
    hf_wet = float(np.sum(np.diff(wet) ** 2))
    assert hf_wet < hf_dry


def test_dsp_deterministic(tmp_path):
    t = write_tone(tmp_path / "t.wav", seconds=1.0)

    def run(dest):
        edl = _single(t)
        e = Dsp(drive=0.4, filter=0.5, chorus=0.3, reverb=0.4).process(edl, _ctx(dest))
        return sf.read(str(e.segments[0].source))[0]

    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    assert np.array_equal(a, b)


def test_dsp_in_chain_via_cli(tmp_path):
    tone = write_tone(tmp_path / "in.wav", seconds=4.0)
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({
        "chain": ["slice", "dsp", "sequence", "splice"],
        "dsp": {"drive": 0.3, "filter": 0.4, "chorus": 0.0, "reverb": 0.3}}))
    out = tmp_path / "out.wav"
    render_one(tone, cfg, out, tmp_path / "scratch")
    assert out.exists() and io.frames_of(out) > 0
