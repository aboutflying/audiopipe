from __future__ import annotations
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from pathlib import Path
import random
from ..segment import EDL


@dataclass
class Context:
    """Per-run services handed to every stage."""
    scratch_dir: Path          # where stages write rendered audio
    rng: random.Random         # seeded; all randomness draws from this
    target_sample_rate: int | None = None  # set if a stage requires a fixed SR
    mono: str = "independent"  # source.mono policy, for stages that materialize


@runtime_checkable
class Stage(Protocol):
    name: str
    def process(self, edl: EDL, ctx: Context) -> EDL: ...
