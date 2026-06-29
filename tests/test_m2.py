from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import soundfile as sf
import yaml

from audiopipe.segment import EDL
from audiopipe.stages.base import Context
from audiopipe.segmenter import Segmenter
from audiopipe.sequencer import Sequencer
from audiopipe.splice import render_edl
from audiopipe import io
from audiopipe.runner import render_one
from .conftest import write_tone


def _ctx(tmp_path, seed=42, mono="sum"):
    return Context(scratch_dir=tmp_path, rng=random.Random(seed), mono=mono)


def _single(tone):
    _, ch, n = io.info(tone)
    return EDL.single(Path(tone), n, 16000, ch, seed=42)


def test_grid_contiguous_no_jitter(tmp_path, tone):
    edl = _single(tone)
    out = Segmenter("grid", amount=0.6, jitter=0.0).process(edl, _ctx(tmp_path))
    segs = out.segments
    assert len(segs) > 1
    assert segs[0].start_frame == 0
    assert segs[-1].end_frame == _single(tone).segments[0].end_frame
    for a, b in zip(segs, segs[1:]):
        assert a.end_frame == b.start_frame      # contiguous
        assert a.n_frames > 0                    # non-empty


def test_jitter_stays_in_bounds(tmp_path, tone):
    full = _single(tone).segments[0]
    out = Segmenter("grid", amount=0.6, jitter=0.8).process(_single(tone), _ctx(tmp_path))
    for s in out.segments:
        assert s.n_frames > 0
        assert full.start_frame <= s.start_frame < s.end_frame <= full.end_frame


def test_shuffle_deterministic_and_seed_varies(tmp_path, tone):
    def order(seed):
        e = Segmenter("grid", 0.6, 0.0).process(_single(tone), _ctx(tmp_path, seed))
        e = Sequencer("shuffle", strength=0.9, drop=0.0).process(e, _ctx(tmp_path, seed))
        return [s.start_frame for s in e.segments]
    assert order(42) == order(42)            # reproducible
    assert order(42) != order(7)             # seed changes arrangement


def test_drop_fraction(tmp_path, tone):
    e = Segmenter("grid", 0.8, 0.0).process(_single(tone), _ctx(tmp_path))
    n = len(e.segments)
    e = Sequencer("as-is", drop=0.25).process(e, _ctx(tmp_path))
    assert len(e.segments) == n - round(0.25 * n)


def test_splice_cut_length(tmp_path, tone):
    e = Segmenter("grid", 0.7, 0.0).process(_single(tone), _ctx(tmp_path))
    total = sum(s.n_frames for s in e.segments)
    out = tmp_path / "spliced.wav"
    render_edl(e, out, join="cut", smear=0.0, mono="sum")
    assert io.frames_of(out) == total


def test_crossfade_no_clipping(tmp_path, tone):
    e = Segmenter("grid", 0.7, 0.0).process(_single(tone), _ctx(tmp_path))
    e = Sequencer("shuffle", 0.7, 0.0).process(e, _ctx(tmp_path))
    out = tmp_path / "xf.wav"
    render_edl(e, out, join="crossfade", smear=0.3, mono="sum")
    data, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert np.max(np.abs(data)) <= 1.0


def _click_energy(path):
    data, _ = sf.read(str(path), dtype="float32")
    return float(np.sum(np.diff(data) ** 2))


def test_zerocross_reduces_click_vs_cut(tmp_path, tone):
    e = Segmenter("grid", 0.85, 0.0).process(_single(tone), _ctx(tmp_path))
    e = Sequencer("shuffle", 0.9, 0.0).process(e, _ctx(tmp_path))
    cut = tmp_path / "cut.wav"
    zc = tmp_path / "zc.wav"
    render_edl(e, cut, join="cut", smear=0.0, mono="sum")
    render_edl(e, zc, join="zerocross", smear=0.0, mono="sum")
    assert _click_energy(zc) < _click_energy(cut)


def test_chain_reorder_no_code_change(tmp_path):
    tone = write_tone(tmp_path / "long.wav", seconds=5.0)
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({"chain": ["sequence", "slice", "splice"]}))
    out = tmp_path / "out.wav"
    render_one(tone, cfg, out, tmp_path / "scratch")
    assert out.exists() and io.frames_of(out) > 0


def test_windowed_peak_memory(tmp_path):
    import tracemalloc
    tone = write_tone(tmp_path / "big.wav", seconds=60.0, sr=44100)
    file_bytes = (tmp_path / "big.wav").stat().st_size
    cfg = tmp_path / "p.yaml"
    cfg.write_text(yaml.safe_dump({"chain": ["slice", "sequence", "splice"]}))
    tracemalloc.start()
    render_one(tone, cfg, tmp_path / "out.wav", tmp_path / "scratch")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < file_bytes / 2   # proves no whole-file load
