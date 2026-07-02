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
from audiopipe.mapping import fx_params
from audiopipe import io
from audiopipe.runner import render_one
from .conftest import write_tone


def _ctx(tmp_path, seed=42, channels="sum"):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    return Context(scratch_dir=Path(tmp_path), seed=seed, channels=channels)


def _single(tone, sr=16000):
    _, ch, n = io.info(tone)
    return EDL.single(Path(tone), n, sr, ch, seed=42)


def test_dsp_params_off_when_zero():
    assert fx_params({"drive": 0, "tone": 0, "chorus": 0, "reverb": 0}) == {}
    p = fx_params({"drive": 1.0, "tone": 0, "chorus": 0, "reverb": 0})
    assert p == {"drive_db": 30.0}


def test_dsp_all_zero_is_passthrough(tmp_path, tone):
    edl = _single(tone)
    src = edl.segments[0].source
    out = Dsp(drive=0, tone=0, chorus=0, reverb=0).process(edl, _ctx(tmp_path))
    assert out.segments[0].source == src          # untouched, no scratch render
    assert out.history[-1]["stage"] == "fx"


def test_dsp_renders_to_scratch_and_transforms(tmp_path, tone):
    edl = _single(tone)
    n_in = edl.segments[0].n_frames
    out = Dsp(drive=0.5, tone=0.6, chorus=0.0, reverb=0.0).process(edl, _ctx(tmp_path))
    seg = out.segments[0]
    assert seg.audio is not None                          # rendered in-memory
    assert "fx" in seg.ops
    assert seg.start_frame == 0 and seg.n_frames == n_in  # filter/drive preserve length
    dry, _ = sf.read(str(tone), always_2d=True)
    assert not np.allclose(dry, seg.audio, atol=1e-3)


def test_filter_reduces_brightness(tmp_path, tone):
    # a dark lowpass should cut high-frequency energy vs the dry signal
    edl = _single(tone)
    out = Dsp(drive=0, tone=0.95, chorus=0, reverb=0).process(edl, _ctx(tmp_path))
    dry, _ = sf.read(str(tone))
    wet = out.segments[0].audio[:, 0]
    hf_dry = float(np.sum(np.diff(dry) ** 2))
    hf_wet = float(np.sum(np.diff(wet) ** 2))
    assert hf_wet < hf_dry


def test_dsp_deterministic(tmp_path):
    t = write_tone(tmp_path / "t.wav", seconds=1.0)

    def run(dest):
        edl = _single(t)
        e = Dsp(drive=0.4, tone=0.5, chorus=0.3, reverb=0.4).process(edl, _ctx(dest))
        return e.segments[0].audio

    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    assert np.array_equal(a, b)


def test_dsp_in_chain_via_cli(tmp_path):
    tone = write_tone(tmp_path / "in.wav", seconds=4.0)
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({
        "chain": ["grain", "fx", "rearrange", "splice"],
        "fx": {"drive": 0.3, "tone": 0.4, "chorus": 0.0, "reverb": 0.3}}))
    out = tmp_path / "out.wav"
    render_one(tone, cfg, out, tmp_path / "scratch")
    assert out.exists() and io.frames_of(out) > 0


def test_master_fx_glue_reverb_tail(tmp_path):
    # master fx reverb pads the output so the tail rings past the end —
    # unlike per-grain fx, where tails truncate at grain boundaries
    tone = write_tone(tmp_path / "in.wav", seconds=2.0)
    base = tmp_path / "dry.yaml"
    base.write_text(yaml.safe_dump({"chain": ["grain", "splice"]}))
    wet = tmp_path / "wet.yaml"
    wet.write_text(yaml.safe_dump({
        "chain": ["grain", "splice"],
        "master": [{"fx": {"reverb": 0.6}}]}))
    render_one(tone, base, tmp_path / "dry.wav", tmp_path / "s1")
    _, edl, _ = render_one(tone, wet, tmp_path / "wet.wav", tmp_path / "s2")
    n_dry = io.frames_of(tmp_path / "dry.wav")
    n_wet = io.frames_of(tmp_path / "wet.wav")
    assert n_wet > n_dry                      # tail pad appended
    y, _ = sf.read(str(tmp_path / "wet.wav"))
    assert np.any(np.abs(y[n_dry:]) > 1e-4)   # and the tail actually rings
    assert any(h["stage"] == "master:fx" for h in edl.history)


def test_master_inline_dials_are_standalone(tmp_path):
    # {fx: {reverb: .5}} in master must NOT inherit the chain fx block's drive
    from audiopipe.pipeline import resolve_config
    cfg = resolve_config({"chain": ["grain", "fx", "splice"],
                          "fx": {"drive": 0.9},
                          "master": [{"fx": {"reverb": 0.5}}]})
    name, params = cfg["master"][0]
    assert name == "fx" and params["reverb"] == 0.5 and params["drive"] == 0.0
    # bare name uses the shared block
    cfg2 = resolve_config({"fx": {"drive": 0.9}, "master": ["fx"]})
    assert cfg2["master"][0][1]["drive"] == 0.9


def test_master_rejects_unknown_pass():
    from audiopipe.pipeline import resolve_config
    with pytest.raises(ValueError, match="unknown master pass"):
        resolve_config({"master": ["vari"]})
