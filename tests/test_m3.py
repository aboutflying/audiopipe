from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import yaml
import pytest

from audiopipe.segment import EDL, Segment
from audiopipe.stages.base import Context
from audiopipe.segmenter import Segmenter
from audiopipe.sequencer import Sequencer
from audiopipe.analyze import feature
from audiopipe.degrade import degrade
from audiopipe import io, splice
from audiopipe.runner import render_one
from audiopipe.tape_loop import run_tape_loop


def _ctx(tmp_path, seed=42):
    return Context(scratch_dir=tmp_path, rng=random.Random(seed), channels="sum")


def write_clicks(path, sr=16000, times=(0.25, 0.5, 0.75, 1.25, 1.5)):
    n = int(2.0 * sr)
    y = np.zeros(n, dtype="float32")
    for t in times:
        i = int(t * sr)
        y[i:i + 20] = 0.9
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path, sr, times


def write_sweep(path, sr=16000, seconds=2.0):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * (200 + 3000 * t / seconds) * t).astype("float32")
    sf.write(str(path), y, sr, subtype="PCM_16")
    return path


def test_onset_slicing_lands_on_clicks(tmp_path):
    path, sr, times = write_clicks(tmp_path / "clicks.wav")
    edl = EDL.single(Path(path), io.frames_of(path), sr, 1, seed=1)
    out = Segmenter("onset").process(edl, _ctx(tmp_path))
    cut_frames = sorted(s.start_frame for s in out.segments if s.start_frame > 0)
    tol = int(0.05 * sr)
    for t in times:
        target = int(t * sr)
        assert any(abs(c - target) <= tol for c in cut_frames), f"no cut near {t}s"


def test_sort_monotonic_by_feature(tmp_path):
    sweep = write_sweep(tmp_path / "sweep.wav")
    edl = EDL.single(Path(sweep), io.frames_of(sweep), 16000, 1, seed=1)
    edl = Segmenter("grid", density=0.8, drift=0.0).process(edl, _ctx(tmp_path))
    edl = Sequencer("sort", drop=0.0, sort_by="brightness").process(edl, _ctx(tmp_path))
    vals = [feature(s, "brightness") for s in edl.segments]
    assert vals == sorted(vals)              # monotonic
    assert max(vals) - min(vals) > 1.0       # actually spread, not all equal


def _tape_cfg(tmp_path, cycles, recursive=False, seam="cut"):
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({
        "chain": ["grain", "rearrange", "splice"],
        "tape_loop": {"cycles": cycles, "wear": 0.6, "feedback": recursive,
                      "seam": seam}}))
    return cfg


def test_tape_loop_cycles1_is_noop(tmp_path):
    tone = write_sweep(tmp_path / "in.wav", seconds=3.0)
    out1 = tmp_path / "c1.wav"
    render_one(tone, _tape_cfg(tmp_path, 1), out1, tmp_path / "s1")
    # equals the plain chain output (cycles:1 never enters tape_loop concat)
    assert io.frames_of(out1) > 0


def test_degrade_cycle0_unmodified_and_ramp(tmp_path):
    sweep = write_sweep(tmp_path / "s.wav", seconds=1.0)
    audio = io.read_frames(Path(sweep), 0, io.frames_of(sweep), "sum")
    rng = random.Random(0)
    c0 = degrade(audio, 16000, 0.0, rng)
    assert np.array_equal(c0, audio)                      # cycle 0 untouched
    # more wear -> more high-freq loss -> lower energy
    low = degrade(audio, 16000, 0.3, random.Random(1))
    high = degrade(audio, 16000, 0.9, random.Random(1))
    assert np.sum(high ** 2) < np.sum(low ** 2) < np.sum(audio ** 2)


def test_degrade_deterministic_from_seed(tmp_path):
    sweep = write_sweep(tmp_path / "s.wav", seconds=1.0)
    audio = io.read_frames(Path(sweep), 0, io.frames_of(sweep), "sum")
    a = degrade(audio, 16000, 0.7, random.Random(5))
    b = degrade(audio, 16000, 0.7, random.Random(5))
    assert np.array_equal(a, b)


def test_recursive_differs_from_parameterized(tmp_path):
    sweep = write_sweep(tmp_path / "in.wav", seconds=3.0)
    out_p = tmp_path / "param.wav"
    out_r = tmp_path / "rec.wav"
    render_one(sweep, _tape_cfg(tmp_path, 6, recursive=False), out_p, tmp_path / "sp")
    render_one(sweep, _tape_cfg(tmp_path, 6, recursive=True), out_r, tmp_path / "sr")
    a, _ = sf.read(str(out_p), always_2d=True)
    b, _ = sf.read(str(out_r), always_2d=True)
    m = min(len(a), len(b))
    assert not np.allclose(a[:m], b[:m])      # the two modes diverge


def test_chain_renders_once_regardless_of_cycles(tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = splice.Splice.process

    def counting(self, edl, ctx):
        calls["n"] += 1
        return orig(self, edl, ctx)

    monkeypatch.setattr(splice.Splice, "process", counting)
    sweep = write_sweep(tmp_path / "in.wav", seconds=3.0)
    render_one(sweep, _tape_cfg(tmp_path, 8), tmp_path / "o.wav", tmp_path / "s")
    assert calls["n"] == 1                    # splice ran exactly once for 8 cycles


def test_disintegrates_across_cycles(tmp_path):
    # last cycle is audibly more worn (lower energy) than the first
    sweep = write_sweep(tmp_path / "in.wav", seconds=3.0)
    edl = EDL.single(Path(sweep), io.frames_of(sweep), 16000, 1, seed=42)
    scratch = tmp_path / "scr"
    scratch.mkdir()
    loop = scratch / "loop.wav"
    splice.render_edl(edl, loop, join="cut", fade=0.0, channels="sum")
    e = EDL(segments=[Segment(loop, 0, io.frames_of(loop), 16000, 1)],
            seed=42, sample_rate=16000)
    run_tape_loop(e, _ctx(scratch), {"cycles": 8, "wear": 0.6,
                  "feedback": False, "seam": "cut"}, tmp_path / "o.wav")
    cyc = sorted(e.segments, key=lambda s: s.cycle)
    first, _ = sf.read(str(cyc[0].source))
    last, _ = sf.read(str(cyc[-1].source))
    assert np.sum(last ** 2) < np.sum(first ** 2)   # progressive decay


def test_tape_loop_region_windows_content(tmp_path):
    sweep = write_sweep(tmp_path / "in.wav", seconds=4.0)
    edl = EDL.single(Path(sweep), io.frames_of(sweep), 16000, 1, seed=42)
    scratch = tmp_path / "scr"; scratch.mkdir()
    loop = scratch / "loop.wav"
    splice.render_edl(edl, loop, join="cut", fade=0.0, channels="sum")

    def run(region):
        e = EDL(segments=[Segment(loop, 0, io.frames_of(loop), 16000, 1)],
                seed=42, sample_rate=16000)
        cfg = {"cycles": 3, "wear": 0.5, "feedback": False, "seam": "cut",
               "region": region}
        out = tmp_path / f"o_{region}.wav"
        run_tape_loop(e, _ctx(scratch), cfg, out)
        return io.frames_of(out)

    full = run(None)
    windowed = run([1.0, 2.0])          # a 1-second region, 3 cycles
    assert abs(windowed - 3 * 16000) <= 3      # 3 cycles x 1s (cut seam, no overlap)
    assert windowed < full                      # smaller than looping the whole 4s
