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
from .dsp import Dsp
from .warp import Warp
from .ott import Ott
from .segment import EDL

# name -> (constructor, config section). A stage with no section takes no args.
STAGES = {
    "passthrough": (Passthrough, None),
    "grain": (Segmenter, "grain"),
    "rearrange": (Sequencer, "rearrange"),
    "splice": (Splice, "splice"),
    "fx": (Dsp, "fx"),
    "vari": (Warp, "vari"),
    "ott": (Ott, "ott"),
}

# Resolved defaults. Top-level and per-section keys are closed: unknown -> error.
DEFAULTS = {
    # Defaults are transparent: every omitted dial is a no-op, so a config only
    # has to declare what it wants to change. The default chain is an identity
    # render (clean grid grains, kept in order, hard-cut back together).
    "seed": 42,
    "source": {"channels": "keep", "sample_rate": "source"},
    "chain": ["grain", "rearrange", "splice"],
    "grain": {"mode": "grid", "density": 0.6, "drift": 0.0},
    "rearrange": {"feel": "as-is", "scramble": 0.0, "drop": 0.0,
                  "sort_by": "brightness"},
    "splice": {"join": "cut", "fade": 0.0, "dropouts": 0.0},
    "fx": {"drive": 0.0, "tone": 0.0, "chorus": 0.0, "reverb": 0.0},
    "vari": {"reverse": 0.0, "speed": 1.0, "wobble": 0.0},
    "ott": {"depth": 0.0, "where": "grain"},
    "tape_loop": {"cycles": 1, "wear": 0.0, "feedback": False,
                  "seam": "cut", "region": None,
                  "hiss": 0.0, "flutter": 0.0, "speed": 1.0, "reverse": False},
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


def load_pipeline(config_path: Path | None, scratch_dir: Path):
    """Return (stages, context, run_config). A missing/None config runs the
    in-code DEFAULTS (slice -> sequence -> splice); presets in config/presets/
    are opt-in overrides passed via -c."""
    raw = {}
    if config_path is not None and Path(config_path).exists():
        raw = yaml.safe_load(Path(config_path).read_text())
    cfg = resolve_config(raw)
    stages = build_stages(cfg)
    ctx = Context(scratch_dir=Path(scratch_dir), rng=random.Random(cfg["seed"]),
                  channels=cfg["source"]["channels"])
    return stages, ctx, cfg


def run_chain(edl: EDL, stages: list[Stage], ctx: Context) -> EDL:
    for stage in stages:
        edl = stage.process(edl, ctx)
    return edl
