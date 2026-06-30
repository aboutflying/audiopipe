from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
import uuid


@dataclass(frozen=True)
class Segment:
    """One slice of audio, by reference. Audio is NOT held here; it is read from
    `source` over [start_frame, end_frame) at render time. This keeps long files
    out of memory until a segment is actually materialized."""
    source: Path
    start_frame: int
    end_frame: int
    sample_rate: int
    channels: int
    # Provenance / op trail. Stages append a short tag describing what they did.
    ops: tuple[str, ...] = ()
    # Tape-loop cycle index (0-based). The tape_loop construct tags each repeated
    # copy with its cycle so a degrade operator can ramp wear across cycles. 0 for
    # all non-looped material.
    cycle: int = 0
    # Stable id so a segment can be traced through the EDL and into the sidecar.
    seg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # In-memory rendered audio (frames, channels). Set by sample-transforming
    # stages (fx/vari/ott) instead of writing scratch; None = read from `source`.
    # Excluded from eq/hash/repr/serialization — `source`+frames are the identity.
    audio: object = field(default=None, compare=False, repr=False)

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame

    def with_op(self, tag: str) -> "Segment":
        return replace(self, ops=self.ops + (tag,))


@dataclass
class EDL:
    """An ordered list of segments plus run-level metadata. The unit every stage
    consumes and produces."""
    segments: list[Segment]
    seed: int
    sample_rate: int
    # Free-form record of stages applied, for the sidecar.
    history: list[dict] = field(default_factory=list)

    @classmethod
    def single(cls, source: Path, n_frames: int, sample_rate: int,
               channels: int, seed: int) -> "EDL":
        """Wrap one continuous file as a one-segment EDL."""
        seg = Segment(source=source, start_frame=0, end_frame=n_frames,
                      sample_rate=sample_rate, channels=channels)
        return cls(segments=[seg], seed=seed, sample_rate=sample_rate)

    def record(self, stage_name: str, params: dict) -> None:
        self.history.append({"stage": stage_name, "params": params,
                             "n_segments": len(self.segments)})
