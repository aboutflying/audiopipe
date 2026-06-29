from __future__ import annotations
from pathlib import Path
import copy
import random
import yaml

from .stages.base import Context, Stage
from .stages.passthrough import Passthrough
from .segmenter import Segmenter
from .sequencer import Sequencer
from .splice import Splice
from .segment import EDL

# name -> (constructor, config section). A stage with no section takes no args.
STAGES = {
    "passthrough": (Passthrough, None),
    "slice": (Segmenter, "slice"),
    "sequence": (Sequencer, "sequence"),
    "splice": (Splice, "splice"),
}

# Resolved defaults. Top-level and per-section keys are closed: unknown -> error.
DEFAULTS = {
    "seed": 42,
    "source": {"mono": "sum", "sample_rate": "source"},
    "chain": ["slice", "sequence", "splice"],
    "slice": {"strategy": "grid", "amount": 0.6, "jitter": 0.3},
    "sequence": {"feel": "shuffle", "strength": 0.7, "drop": 0.1,
                 "sort_by": "brightness"},
    "splice": {"join": "crossfade", "smear": 0.2},
    "tape_loop": {"cycles": 1, "evolve": 0.4, "recursive": False,
                  "seam": "crossfade"},
}


def _merge(defaults: dict, user: dict, where: str) -> dict:
    for k in user:
        if k not in defaults:
            raise ValueError(f"unknown config key {where}{k!r}")
    out = {}
    for k, dv in defaults.items():
        if isinstance(dv, dict):
            out[k] = _merge(dv, user.get(k, {}), f"{where}{k}.")
        else:
            out[k] = user.get(k, dv)
    return out


def resolve_config(raw: dict) -> dict:
    cfg = _merge(DEFAULTS, raw or {}, "")
    for name in cfg["chain"]:
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r} in chain")
    return cfg


def build_stages(cfg: dict) -> list[Stage]:
    stages = []
    for name in cfg["chain"]:
        ctor, section = STAGES[name]
        stages.append(ctor(**cfg[section]) if section else ctor())
    return stages


def load_pipeline(config_path: Path, scratch_dir: Path):
    """Return (stages, context, run_config)."""
    raw = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    cfg = resolve_config(raw)
    stages = build_stages(cfg)
    ctx = Context(scratch_dir=Path(scratch_dir), rng=random.Random(cfg["seed"]),
                  mono=cfg["source"]["mono"])
    return stages, ctx, cfg


def run_chain(edl: EDL, stages: list[Stage], ctx: Context) -> EDL:
    for stage in stages:
        edl = stage.process(edl, ctx)
    return edl
