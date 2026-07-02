from __future__ import annotations
from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field
from pathlib import Path
import random
from ..segment import EDL


@dataclass
class Context:
    """Per-run services handed to every stage."""
    scratch_dir: Path          # where stages write rendered audio
    seed: int = 0              # master seed; stages draw via rng_for(name)
    target_sample_rate: int | None = None  # set if a stage requires a fixed SR
    channels: str = "keep"     # source.channels policy, for stages that materialize
    _rngs: dict = field(default_factory=dict, repr=False)

    def rng_for(self, name: str) -> random.Random:
        """Per-stage RNG derived from (seed, name). Each stage's randomness is an
        independent stream, so changing one stage's dials (which changes how many
        draws it makes) never rerolls another stage's choices — a found
        arrangement survives tweaking a single dial. Cached, so duplicate chain
        entries of the same stage continue one stream deterministically."""
        if name not in self._rngs:
            self._rngs[name] = random.Random(f"{self.seed}:{name}")
        return self._rngs[name]


@runtime_checkable
class Stage(Protocol):
    name: str
    def process(self, edl: EDL, ctx: Context) -> EDL: ...
