from __future__ import annotations
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import soundfile as sf
import pytest
import yaml

from audiopipe.segment import EDL
from audiopipe.pipeline import resolve_config, build_stages, load_pipeline
from audiopipe.queue import Queue
from audiopipe import io
from audiopipe.runner import render_one, process_inbox
from audiopipe.sidecar import sha256


def test_edl_single(tone):
    _, _, n = io.info(tone)
    edl = EDL.single(Path(tone), n, 16000, 1, seed=1)
    assert len(edl.segments) == 1
    assert edl.segments[0].n_frames == n
    assert edl.segments[0].start_frame == 0 and edl.segments[0].end_frame == n


def test_loader_rejects_unknown_key():
    with pytest.raises(ValueError, match="unknown config key"):
        resolve_config({"seed": 1, "nope": True})
    with pytest.raises(ValueError, match="unknown config key"):
        resolve_config({"slice": {"bogus": 1}})


def test_loader_defaults_and_order():
    cfg = resolve_config({"chain": ["sequence", "slice"]})
    assert cfg["seed"] == 42
    assert cfg["slice"]["strategy"] == "grid"
    stages = build_stages(cfg)
    assert [s.name for s in stages] == ["sequence", "slice"]


def test_queue_claim_atomic_idempotent(tmp_path, tone):
    q = Queue(tmp_path)
    dst = q.inbox / "a.wav"
    dst.write_bytes(Path(tone).read_bytes())
    claimed = q.claim(dst)
    assert claimed is not None and claimed.parent == q.working
    # second claim of the same (now-moved) path fails cleanly
    assert q.claim(dst) is None


def _passthrough_cfg(tmp_path) -> Path:
    p = tmp_path / "pipeline.yaml"
    p.write_text(yaml.safe_dump({"seed": 7, "source": {"channels": "sum"},
                                 "chain": ["passthrough"]}))
    return p


def test_passthrough_bit_identical(tmp_path, tone):
    cfg = _passthrough_cfg(tmp_path)
    out = tmp_path / "out.wav"
    render_one(tone, cfg, out, tmp_path / "scratch")
    a, _ = sf.read(str(tone), dtype="float32", always_2d=True)
    b, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-4)


@pytest.mark.skipif(shutil.which("afconvert") is None, reason="needs macOS afconvert")
def test_m4a_transcoded_on_fetch(tmp_path, tone):
    # encode the tone to m4a (libsndfile can't read it), then process it
    m4a = tmp_path / "clip.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(tone), str(m4a)],
                   check=True, capture_output=True)
    with pytest.raises(sf.LibsndfileError):
        sf.info(str(m4a))                       # confirm it's genuinely unreadable
    cfg = _passthrough_cfg(tmp_path)
    out = tmp_path / "out.wav"
    render_one(m4a, cfg, out, tmp_path / "scratch")
    assert out.exists() and io.frames_of(out) > 0
    # sidecar hashes the original m4a, not the decoded wav
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["input"] == str(m4a)
    assert side["input_sha256"] == sha256(m4a)


def test_process_drains_inbox_and_sidecar(tmp_path, tone):
    cfg = _passthrough_cfg(tmp_path)
    q = Queue(tmp_path / "work")
    (q.inbox / "song.wav").write_bytes(Path(tone).read_bytes())
    outs = process_inbox(tmp_path / "work", cfg)
    assert len(outs) == 1 and outs[0].exists()
    # moved inbox -> done
    assert not list(q.inbox.iterdir())
    assert (q.done / "song.wav").exists()
    # sidecar round-trips
    side = json.loads(outs[0].with_suffix(".json").read_text())
    assert side["input_sha256"] == sha256(q.done / "song.wav")
    assert len(side["edl"]["segments"]) == 1
