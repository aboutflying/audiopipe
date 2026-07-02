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

        # Post-chain: render the chain to one "master" file, then apply the
        # `master:` passes in configured order (fx = glue, ott = bus comp,
        # tape = physical medium — by convention last, so tape character is
        # never fed into the compressor).
        master = _render_master(edl, ctx, scratch_dir)
        for name, params in cfg["master"]:
            if name == "fx":
                from .dsp import fx_file
                fx_file(master, params)
                edl.record("master:fx", params)
            elif name == "ott":
                if params["depth"] > 0:
                    from .ott import ott_file
                    ott_file(master, params["depth"])
                edl.record("master:ott", params)
            elif name == "tape":
                from .tape import run_tape
                run_tape(edl, ctx, params, master, master)  # records itself
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


def watch(work_root: Path, config_path: Path, interval: float = 2.0,
          once: bool = False) -> None:
    """The always-on worker loop (the reconcile loop from the original spec):
    poll inbox/, render whatever lands there, repeat. Ctrl-C to stop.
    ponytail: stdlib polling; swap in watchdog/FSEvents if latency ever matters."""
    import time
    print(f"watching {Path(work_root) / 'inbox'} (every {interval:g}s, Ctrl-C to stop)")
    while True:
        try:
            for out in process_inbox(work_root, config_path):
                print(out, flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the daemon alive
            print(f"error: {exc!r}", flush=True)
        if once:
            return
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            return


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
