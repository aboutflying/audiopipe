from __future__ import annotations
from pathlib import Path
import shutil

from . import io, sidecar
from .segment import EDL
from .pipeline import load_pipeline, run_chain
from .splice import render_edl
from .queue import Queue


def _resolve_sr(source_sr: int, want) -> int:
    return source_sr if want == "source" else int(want)


def render_one(input_path: Path, config_path: Path, out_path: Path,
               scratch_dir: Path) -> tuple[Path, EDL, dict]:
    """Run the full chain on one file, write rendered output to out_path.
    Returns (out_path, final_edl, run_config)."""
    input_path = Path(input_path)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Transcode non-libsndfile inputs (M4A/AAC) to a readable WAV in scratch;
        # audio is read from `readable`, but the sidecar still hashes the original.
        readable = io.ensure_readable(input_path, scratch_dir)
        src_sr, ch, n = io.info(readable)

        stages, ctx, cfg = load_pipeline(config_path, scratch_dir)
        sr = _resolve_sr(src_sr, cfg["source"]["sample_rate"])
        edl = EDL.single(readable, n, sr, ch, cfg["seed"])

        edl = run_chain(edl, stages, ctx)

        # Post-chain signal flow, in order:
        #   1. render the chain to one "master" file (the clean collage)
        #   2. OTT master compression (whole-output)
        #   3. tape: physical medium (hiss/wear/flutter/loop) applied LAST, so
        #      tape character is never fed into OTT.
        master = _render_master(edl, ctx, scratch_dir)

        ott = cfg["ott"]
        if ott["where"] == "output" and ott["depth"] > 0:
            from .ott import ott_file
            ott_file(master, ott["depth"])

        tl = cfg["tape_loop"]
        if (tl["cycles"] > 1 or tl["hiss"] > 0 or tl["flutter"] > 0
                or tl["speed"] != 1.0 or tl["reverse"]):
            from .tape_loop import run_tape_loop
            run_tape_loop(edl, ctx, tl, master, out_path)
        else:
            shutil.copy2(master, out_path)

        sidecar.write_success(out_path, input_path=input_path, config=cfg, edl=edl)
        return out_path, edl, cfg
    finally:
        # scratch holds only intermediate renders; output + sidecar are already
        # written outside it. Always discard, even on failure.
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _render_master(edl: EDL, ctx, scratch_dir: Path) -> Path:
    """Render the chain output to a single scratch file (the clean pre-tape mix).
    If splice already rendered to a scratch file, reuse it; else cut-concatenate."""
    segs = edl.segments
    if len(segs) == 1 and segs[0].audio is None \
            and scratch_dir in Path(segs[0].source).parents:
        return Path(segs[0].source)
    master = scratch_dir / "master.wav"
    render_edl(edl, master, join="cut", fade=0.0, channels=ctx.channels)
    return master


def process_inbox(work_root: Path, config_path: Path) -> list[Path]:
    """Drain inbox/: claim each file, render, move to done/ (or failed/)."""
    q = Queue(work_root)
    scratch = Path(work_root) / "scratch"
    outputs = []
    for path in q.list_new():
        claimed = q.claim(path)
        if claimed is None:
            continue
        out_path = q.outbox / (claimed.stem + ".wav")
        try:
            render_one(claimed, config_path, out_path, scratch)
            q.finish(claimed)
            outputs.append(out_path)
        except Exception as exc:  # noqa: BLE001 - record and preserve input
            failed = q.fail(claimed)
            try:
                cfg = load_pipeline(config_path, scratch)[2]
            except Exception:
                cfg = {}
            sidecar.write_failure(q.failed / (failed.stem + ".json"),
                                  input_path=failed, config=cfg, exc=exc)
    return outputs
