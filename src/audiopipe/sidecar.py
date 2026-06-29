from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import traceback as _tb
from datetime import datetime, timezone

from . import __version__
from .segment import EDL


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seg_dict(seg) -> dict:
    d = asdict(seg)
    d["source"] = str(seg.source)
    return d


def _edl_dict(edl: EDL) -> dict:
    return {"seed": edl.seed, "sample_rate": edl.sample_rate,
            "history": edl.history, "segments": [_seg_dict(s) for s in edl.segments]}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_success(out_path: Path, *, input_path: Path, config: dict, edl: EDL) -> Path:
    side = Path(out_path).with_suffix(".json")
    side.write_text(json.dumps({
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "config": config,
        "edl": _edl_dict(edl),
        "audiopipe_version": __version__,
        "timestamp": _now(),
    }, indent=2))
    return side


def write_failure(side_path: Path, *, input_path: Path, config: dict, exc: BaseException) -> Path:
    side = Path(side_path).with_suffix(".json")
    side.write_text(json.dumps({
        "input": str(input_path),
        "config": config,
        "error": repr(exc),
        "traceback": "".join(_tb.format_exception(type(exc), exc, exc.__traceback__)),
        "audiopipe_version": __version__,
        "timestamp": _now(),
    }, indent=2))
    return side
