from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import yaml

from audiopipe.segment import EDL
from audiopipe.stages.base import Context
from audiopipe.warp import Warp
from audiopipe.segmenter import Segmenter
from audiopipe import io
from audiopipe.runner import render_one
from .conftest import write_tone


def _ctx(tmp_path, seed=42, channels="sum"):
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    return Context(scratch_dir=Path(tmp_path), rng=random.Random(seed), channels=channels)


def _single(tone, sr=16000):
    _, ch, n = io.info(tone)
    return EDL.single(Path(tone), n, sr, ch, seed=42)


def test_reverse_flips_audio(tmp_path, tone):
    edl = _single(tone)
    dry = io.read_frames(Path(tone), 0, edl.segments[0].n_frames, "sum")
    out = Warp(reverse=1.0).process(edl, _ctx(tmp_path))
    wet = io.read_frames(out.segments[0].source, 0, out.segments[0].n_frames, "sum")
    assert np.allclose(wet, dry[::-1], atol=1e-4)
    assert "rev" in out.segments[0].ops


def test_speed_up_halves_length(tmp_path, tone):
    edl = _single(tone)
    n_in = edl.segments[0].n_frames
    out = Warp(speed=2.0).process(edl, _ctx(tmp_path))
    assert abs(out.segments[0].n_frames - round(n_in / 2)) <= 1


def test_speed_down_doubles_length(tmp_path, tone):
    edl = _single(tone)
    n_in = edl.segments[0].n_frames
    out = Warp(speed=0.5).process(edl, _ctx(tmp_path))
    assert abs(out.segments[0].n_frames - round(n_in / 0.5)) <= 1


def test_identity_is_reference_only(tmp_path, tone):
    edl = _single(tone)
    src = edl.segments[0].source
    out = Warp(reverse=0.0, speed=1.0, wobble=0.0).process(edl, _ctx(tmp_path))
    assert out.segments[0].source == src          # untouched, no scratch render


def test_partial_reverse_deterministic(tmp_path):
    tone = write_tone(tmp_path / "t.wav", seconds=3.0)

    def run(dest):
        e = Segmenter("grid", 0.8, 0.0).process(_single(tone), _ctx(dest, seed=1))
        e = Warp(reverse=0.5, wobble=0.2).process(e, _ctx(dest, seed=7))
        return [s.n_frames for s in e.segments], sum("rev" in s.ops for s in e.segments)

    a = run(tmp_path / "a")
    b = run(tmp_path / "b")
    assert a == b                                  # reproducible from seed
    assert 0 < a[1] < 100                          # some but not all reversed


def test_warp_in_chain_via_cli(tmp_path):
    tone = write_tone(tmp_path / "in.wav", seconds=4.0)
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({
        "chain": ["grain", "vari", "rearrange", "splice"],
        "vari": {"reverse": 0.4, "speed": 1.3, "wobble": 0.1}}))
    out = tmp_path / "out.wav"
    render_one(tone, cfg, out, tmp_path / "scratch")
    assert out.exists() and io.frames_of(out) > 0
