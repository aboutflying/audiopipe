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

        # tape_loop (M3): post-chain construct, render-once then degrade per cycle.
        if cfg["tape_loop"]["cycles"] > 1:
            from .tape_loop import run_tape_loop
            run_tape_loop(edl, ctx, cfg["tape_loop"], out_path)
        else:
            _finalize(edl, ctx, out_path, scratch_dir)

        sidecar.write_success(out_path, input_path=input_path, config=cfg, edl=edl)
        return out_path, edl, cfg
    finally:
        # scratch holds only intermediate renders; output + sidecar are already
        # written outside it. Always discard, even on failure.
        shutil.rmtree(scratch_dir, ignore_errors=True)


def _finalize(edl: EDL, ctx, out_path: Path, scratch_dir: Path) -> None:
    """Produce the output wav. If the chain already rendered to scratch (splice),
    copy that file; otherwise concatenate segments with a hard cut."""
    segs = edl.segments
    if len(segs) == 1 and scratch_dir in Path(segs[0].source).parents:
        shutil.copy2(segs[0].source, out_path)
    else:
        render_edl(edl, out_path, join="cut", smear=0.0, channels=ctx.channels)


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
