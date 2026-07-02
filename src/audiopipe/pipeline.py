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
    "ott": {"depth": 0.0},
    "tape": {"cycles": 1, "wear": 0.0, "feedback": False,
             "seam": "cut", "region": None,
             "hiss": 0.0, "flutter": 0.0, "speed": 1.0, "reverse": False},
    # Whole-output passes applied to the rendered mix, in order (like `chain`,
    # order is the composition). Entries: a name, or {name: {dials}} for
    # standalone settings independent of the shared section block.
    "master": [],
}

# Passes allowed in `master`. fx = glue effects (reverb tails ring past cuts),
# ott = bus compression, tape = the physical medium (applied last by convention).
MASTER_PASSES = ("fx", "ott", "tape")


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


def _resolve_master(entries, cfg: dict) -> list:
    """Normalize `master` entries to [name, params] pairs. A bare name uses the
    shared section block; {name: {dials}} is standalone over that section's
    DEFAULTS (transparent: unset dials are off)."""
    out = []
    for e in entries:
        if isinstance(e, str):
            name, overrides = e, None
        elif isinstance(e, dict) and len(e) == 1:
            name, overrides = next(iter(e.items()))
        else:
            raise ValueError(f"master entry must be a pass name or "
                             f"{{name: {{dials}}}}, got {e!r}")
        if name not in MASTER_PASSES:
            raise ValueError(f"unknown master pass {name!r}; "
                             f"pick from {list(MASTER_PASSES)}")
        params = (cfg[name] if overrides is None
                  else _merge(DEFAULTS[name], overrides, f"master.{name}."))
        out.append([name, params])
    return out


# Enumerated values, validated at load so a typo fails loud instead of deep in
# a render (e.g. `channels: independent` used to die inside splice).
_ENUMS = {
    ("source", "channels"): ("sum", "left", "keep"),
    ("grain", "mode"): ("grid", "random", "onset", "silence"),
    ("rearrange", "feel"): ("as-is", "shuffle", "reverse", "sort"),
    ("rearrange", "sort_by"): ("brightness", "loudness", "duration"),
    ("splice", "join"): ("cut", "zerocross", "crossfade"),
    ("tape", "seam"): ("cut", "zerocross", "crossfade"),
}


def resolve_config(raw: dict) -> dict:
    cfg = _merge(DEFAULTS, raw or {}, "")
    for name in cfg["chain"]:
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r} in chain")
    for (sec, key), allowed in _ENUMS.items():
        if cfg[sec][key] not in allowed:
            raise ValueError(f"{sec}.{key}: {cfg[sec][key]!r} not allowed; "
                             f"pick one of {list(allowed)}")
    cfg["master"] = _resolve_master(cfg["master"], cfg)
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
    ctx = Context(scratch_dir=Path(scratch_dir), seed=cfg["seed"],
                  channels=cfg["source"]["channels"])
    return stages, ctx, cfg


def run_chain(edl: EDL, stages: list[Stage], ctx: Context) -> EDL:
    for stage in stages:
        edl = stage.process(edl, ctx)
    return edl
