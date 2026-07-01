# audiopipe: Milestones 1 to 3 Spec

A self-hosted audio pipeline that watches a folder, slices/resequences/processes
audio tape-collage style, and writes results back with reproducible metadata.
This document specifies the foundation and core creative engine: the canonical data
model, the config contract, the stage protocol, the directory queue, and milestones
1 through 3. Build it in order. Each milestone must run and pass tests before the
next starts.

## Mental model

The cloud share is transport only (a job inbox and outbox). The real work happens
in an always-on local worker. This is a reconcile loop: a watcher detects new files
(desired state), a processor renders them (drives toward it), output is published.

The canonical intermediate representation is an **EDL** (edit decision list): an
ordered list of `Segment`s. A continuous file is just an EDL with one segment. There
is no separate "continuous mode." Every stage is `EDL -> EDL`. The chain order is
declared in config, so `slice -> dsp -> sequence` versus `sequence -> slice -> dsp`
is a config reorder, never a code branch.

Imperfection is a set of dials, not an accident. Jitter, tape-loop wear, and hard-cut
joins are things you turn up. Everything is seeded, so a happy accident is reproducible
and then tweakable.

## Scope

In scope for M1+M2:
- Segment / EDL dataclasses (the IR)
- Config loader: YAML -> ordered chain of stage objects
- Stage protocol
- Directory-based queue state machine
- Local storage backend behind a protocol (so S3/Dropbox/iCloud swap in later)
- Windowed audio IO (do not load long files whole)
- Passthrough stage (M1)
- Segmenter, sequencer, splice (M2)
- Sidecar metadata that records the full EDL and all seeds

Out of scope for M1+M2 (later milestones, do not build yet):
- Analysis stage B (librosa features, onset slicing, feature-sorted sequencing) -> M3
- Tape-loop construct (render-once, degrade-per-cycle) -> M3
- DSP stage A (pedalboard) -> M4
- Watcher daemon and worker loop -> M4
- Non-local storage backends (define the protocol, implement only local)

Deferred indefinitely (optional addon, not on the current roadmap):
- RAVE / neural resynthesis stage C. The architecture leaves room for it (any
  sample-transforming stage writes to scratch and returns segments pointing at it),
  but do NOT build it now and do NOT pull torch. When it lands it is one more
  `EDL -> EDL` stage plus sample-rate/mono discipline in `io.py`. Nothing in M1-M4
  should depend on it existing.

## Canonical IR: `src/audiopipe/segment.py`

This is the contract everything else implements against. Build it first and freeze it.

```python
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
    # all non-looped material. Added now (before M3) so it never has to be
    # retrofitted through every stage later.
    cycle: int = 0
    # Stable id so a segment can be traced through the EDL and into the sidecar.
    seg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

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
```

Design notes for the implementer:
- `Segment` is frozen and reference-only. Slicing produces new `Segment`s pointing
  at the same `source` with different frame bounds. No audio is copied during
  slicing or sequencing.
- A stage that actually transforms samples (DSP, and any future resynthesis stage)
  must write its output to a scratch file and return segments pointing at that file.
  Define a scratch dir in `Context` (below). Do not hold rendered audio in the EDL.
- `ops` and `history` exist so the sidecar can reproduce and so a human can read what
  happened. Keep tags short, e.g. `"slice:onset"`, `"seq:shuffle"`.

## Stage protocol: `src/audiopipe/stages/base.py`

```python
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

@runtime_checkable
class Stage(Protocol):
    name: str
    def process(self, edl: EDL, ctx: Context) -> EDL: ...
```

Every stage takes an EDL, returns an EDL, and draws any randomness from `ctx.rng`
so the global seed governs the whole run. Stages must be deterministic given the
seed. A stage records itself via `edl.record(self.name, params)` before returning.

## Config contract: `config/pipeline.yaml`

Coarse dials only. Roughly 0-to-1 where it makes sense. The loader expands these
into concrete values; fine detail lives in code, not here. Unknown keys are an
error (fail loud), missing keys take documented defaults.

```yaml
seed: 42                    # global determinism; change to roll a new variant

source:
  mono: sum                # sum | left | independent
  sample_rate: source      # "source" (keep input rate) or an int like 44100

# Order IS the composition. Reorder to change slice<->process<->sequence order.
# A stage absent from this list is skipped for the run.
chain:
  - slice
  - sequence
  - splice

slice:
  strategy: grid           # M2: grid | random   (onset/silence come with B)
  amount: 0.6              # coarse density: low = long slices, high = chopped
  jitter: 0.3              # imperfection: randomizes boundaries (0 = clean)

sequence:
  feel: shuffle            # M2: shuffle | reverse | as-is   (sort comes with B)
  strength: 0.7            # how far from original order it strays
  drop: 0.1                # fraction of slices discarded (0 = keep all)

splice:
  join: crossfade          # cut | zerocross | crossfade
  smear: 0.2               # crossfade length as a coarse dial
```

Loader contract (`src/audiopipe/pipeline.py`):
- Parse YAML, validate against a schema, reject unknown keys.
- Build an ordered `list[Stage]` from `chain`, pulling each stage's config block.
- A `STAGES` registry maps name -> stage constructor: `{"slice": Segmenter,
  "sequence": Sequencer, "splice": Splice, "passthrough": Passthrough}`.
- Construct the seeded `random.Random(seed)` once and put it in `Context`.
- Return `(stages, context, run_config)` so the runner can execute and the sidecar
  can record the exact config used.

Note on the deferred escape hatch: sequencing strategies are named feels, not
scriptable in YAML. Novel resequencing (markov, custom interleave) is a new named
strategy in `sequencer.py`. No `custom:` hook for now. Revisit only if a real need
appears.

## Directory queue: `src/audiopipe/queue.py`

A file moves through directories as its state changes. This is the whole state
machine. It is debuggable by `ls` and survives restarts.

```
work/
  inbox/      # new files land here (M1: dropped manually; later: watcher)
  working/    # claimed, mid-render
  done/       # rendered successfully (output + sidecar alongside, or in outbox/)
  failed/     # errored; original preserved for inspection
```

Rules:
- Claim by atomic move inbox -> working (rename is atomic on a single filesystem).
  This gives idempotency and prevents double processing.
- On success move to done and write output + sidecar.
- On failure move to failed and write an error sidecar (exception, traceback,
  config used). Never delete the input.
- Processing must be idempotent: a given input + config + seed produces the same
  output, so re-running a failed file is safe.

## Storage protocol: `src/audiopipe/storage/`

Define the protocol now, implement only local. This is the seam that lets S3 /
Dropbox / iCloud drop in later without touching the pipeline.

```python
from typing import Protocol
from pathlib import Path

class StorageBackend(Protocol):
    def list_new(self) -> list[Path]: ...
    def fetch(self, path: Path) -> Path: ...   # ensure local & readable; iCloud
                                               # impl will brctl-download here
    def publish(self, local_path: Path) -> None: ...
```

Local backend: `list_new` reads `inbox/`, `fetch` is a no-op (already local),
`publish` writes to `outbox/` or `done/`.

## Audio IO: `src/audiopipe/io.py`

- Use `soundfile` for windowed reads: `sf.SoundFile(path)` then `seek` + `read(n)`.
  Never read a long file whole.
- `frames_of(path) -> int`, `info(path) -> (sample_rate, channels, n_frames)`.
- Mono policy from `source.mono`: sum, take-left, or keep independent.
- Block-write output rather than accumulating one giant array.
- Resample helper (for any stage that needs a fixed sample rate); not exercised in
  M1/M2 but stub it so the seam exists.

## Sidecar: `src/audiopipe/sidecar.py`

Write a JSON file alongside each output recording:
- input path + content hash (sha256 of the source bytes)
- the full resolved run config (including seed)
- the final EDL (every segment: source, start, end, ops)
- timestamp, audiopipe version
On failure, write the same plus exception and traceback.

This makes any collage reproducible and hand-editable: re-render exactly, mutate one
param, or edit the EDL and re-run.

---

## Milestone 1: skeleton + passthrough

Goal: prove the plumbing end to end with zero DSP.

Build:
- `segment.py` (Segment, EDL) per the contract above.
- `stages/base.py` (Stage protocol, Context).
- `stages/passthrough.py` (returns the EDL unchanged, records itself).
- `io.py` (windowed read, info, mono policy, block write).
- `queue.py` (directory state machine, atomic claim).
- `storage/base.py` + `storage/local.py`.
- `pipeline.py` (config loader, stage registry, runner).
- `sidecar.py`.
- `cli.py`: `audiopipe run <file>` runs the chain on one file;
  `audiopipe process` drains `inbox/`.

A passing M1 means: drop a wav in `inbox/`, run `audiopipe process`, and the same
audio appears in `outbox/`/`done/` with a correct sidecar (hash, config, one-segment
EDL), the file having moved inbox -> working -> done. `chain: [passthrough]`.

Tests:
- EDL.single wraps a file as one segment with correct frame count.
- Loader rejects unknown keys; applies defaults; builds the chain in declared order.
- Queue claim is atomic and idempotent (re-running a done file is a no-op or a clean
  re-render).
- Passthrough output is bit-identical to input (within mono-policy expectations).
- Sidecar round-trips: hash matches, EDL serializes and deserializes.

## Milestone 2: collage core

Goal: tape-collage output from slicing and resequencing alone, before any DSP or RAVE
exists. This is the aesthetic heart, so it comes early.

Build:
- `segmenter.py`: `grid` (fixed length from `amount`) and `random` (random lengths
  bounded by `amount`), both honoring `jitter` to randomize boundaries. 1 segment in,
  N out. All randomness from `ctx.rng`.
- `sequencer.py`: `shuffle` (seeded, `strength` controls displacement), `reverse`,
  `as-is`. `drop` discards a fraction of segments. N in, <=N out, reordered.
- `splice.py`: render the EDL to continuous audio. Join policies: `cut` (hard,
  clicky, allowed on purpose), `zerocross` (snap cut points to zero crossings),
  `crossfade` (equal-power, length from `smear`). This is where segments get
  materialized via windowed reads and written out block by block.
- Wire `slice`, `sequence`, `splice` into the registry and let the chain order them.

A passing M2 means: with `chain: [slice, sequence, splice]`, a long input is sliced,
resequenced, and spliced into a recognizably tape-collage output, fully reproducible
from the seed, with the EDL in the sidecar reflecting every cut and reorder. Changing
the seed produces a different but equally reproducible arrangement. Reordering the
chain (e.g. putting `sequence` before `slice`) runs without code changes.

Tests:
- Grid slicing of an N-frame file at a given amount yields the expected segment count
  and contiguous, non-overlapping bounds when jitter is 0.
- Jitter > 0 perturbs boundaries but stays within source bounds and never produces
  negative-length segments.
- Shuffle with a fixed seed is deterministic and reproducible; a different seed gives
  a different order.
- Drop removes the right fraction.
- Splice output length equals the sum of segment lengths (for `cut`), within a known
  tolerance for crossfade overlap.
- Crossfade introduces no clipping; cut is allowed to click; zerocross reduces click
  energy versus cut on a test tone.
- Full chain on a multi-minute file holds peak memory well under the file size
  (proves windowed reads, not whole-file load).

## Milestone 3: analysis (B) and tape loop

Two additions. Analysis makes the collage feel intentional rather than random. The
tape loop adds evolving repetition. Neither needs torch.

### Part A: analysis stage B (`src/audiopipe/analyze.py`, `mapping.py`)

Build:
- `analyze.py`: extract per-segment and whole-file features with librosa. At minimum
  onset envelope, onset times, RMS/loudness, spectral centroid (brightness). Attach a
  small feature dict to each segment (store in `ops`/a parallel map, not as raw audio).
- Feed the segmenter: add `onset` and `silence` slicing strategies that cut on detected
  onsets / silence boundaries instead of a grid.
- Feed the sequencer: add `sort` (order by a feature, e.g. brightness or loudness) and
  feature-weighted shuffle/drop (e.g. drop the quietest fraction, weight order by RMS).
- `mapping.py`: coarse dial -> concrete value expansion lives here. This is where
  `amount: 0.7` becomes actual grain sizes and `sort_by: brightness` resolves to a
  feature key. Keep curves here, out of the stages.

Config additions:
```yaml
slice:
  strategy: onset          # now also: onset | silence
sequence:
  feel: sort               # now also: sort
  sort_by: brightness      # brightness | loudness | duration
```

A passing B means: onset-driven slicing lands cuts on transients (audibly different
from grid), and `feel: sort` with `sort_by: brightness` produces a spectrally graded
arrangement, both reproducible from the seed.

### Part B: tape loop (`src/audiopipe/tape_loop.py`)

This is a post-chain runner construct, NOT a stage in `chain:`. It runs after the
chain produces one rendered "loop content" file, so the expensive work (slicing,
sequencing, splicing, and any future DSP/resynthesis) happens exactly once. Only the
cheap degrade operator iterates per cycle. This is both the compute win and the
authentic tape behavior: a real loop's content is fixed at record time; what changes
each revolution is the medium wearing.

Flow:
1. Run the chain -> splice produces one rendered loop file in scratch (`loop.wav`).
2. `tape_loop` builds `cycles` copies, each a `Segment` referencing `loop.wav` tagged
   with its `cycle` index (0-based). Reference-only, so no audio is copied here.
3. The degrade operator produces each cycle's audio as a function of wear:
   - `recursive: false` (default, cheap): degrade at cycle N is `f(N)` applied to the
     original `loop.wav` in a single pass per cycle. Cycles are independent, computed
     directly from the cycle index. Good for lowpass roll-off, level loss, added
     noise floor, dropouts.
   - `recursive: true`: cycle N+1 degrades cycle N's actual output. Required only for
     non-separable effects that genuinely self-feed (saturation, feedback smear).
     Still cheap because no chain/resynthesis re-runs, only the light degrade op.
4. Concatenate the cycles with the `seam` join (reuse splice join policies). The seam
   is deliberately separate from inter-segment `splice.join` so the loop point can
   click (hard `cut`) while internal cuts stay smooth, true to how a physical splice
   sounds each revolution.

Config (top-level, mirrors its position around the chain):
```yaml
tape_loop:
  cycles: 8            # 1 = off (plain single render)
  evolve: 0.4          # accumulated wear per cycle (0 = faithful repeats)
  recursive: false     # false = parameterized f(cycle) [cheap]; true = iterate op
  seam: crossfade      # join at the loop point: cut | zerocross | crossfade
```

Degrade operator (`src/audiopipe/degrade.py`): a small set of wear effects scaled by
`evolve * (cycle / max(cycles-1, 1))`, so wear ramps from none at cycle 0 to full at
the last cycle. Start with lowpass roll-off, gain attenuation, and random dropouts.
All randomness from `ctx.rng` so the whole loop reproduces from the master seed.

A passing tape loop means: an 8-cycle render audibly disintegrates across its length
(progressive dulling/dropouts) rather than repeating identically, reproduces exactly
from the seed, records `cycles`/`evolve`/per-cycle wear and the full EDL in the
sidecar, and bumping `cycles` from 8 to 9 leaves cycles 0-7 unchanged. With
`cycles: 1` the output equals the plain M2 collage.

Tests:
- librosa onset slicing places boundaries at detected onsets within tolerance on a
  click-track fixture.
- `feel: sort` orders segments monotonically by the chosen feature.
- tape_loop with `cycles: 1` is a no-op (output equals chain output).
- Parameterized degrade is deterministic from the seed; wear at cycle N matches the
  ramp formula; cycle 0 is unmodified.
- Recursive degrade with a fixed seed reproduces; output differs from parameterized
  on a self-feeding effect (proves the two modes diverge as intended).
- The chain runs once regardless of `cycles` (assert render-once: count splice
  invocations == 1 for any cycle count).

Dependencies added at M3: `librosa`, `scipy`. Still no torch.

## Build order summary

1. `segment.py` (freeze the IR, including the `cycle` field)
2. `stages/base.py`, `Context`
3. `io.py`, `queue.py`, `storage/`
4. `pipeline.py` loader + runner, `sidecar.py`, `passthrough`, `cli.py`  -> M1 done
5. `segmenter.py`, `sequencer.py`, `splice.py`  -> M2 done
6. `analyze.py`, `mapping.py` (onset/silence slicing, sort/weighted sequencing)
7. `degrade.py`, `tape_loop.py` (render-once, degrade-per-cycle)  -> M3 done

Dependencies by milestone: M1+M2 need `soundfile`, `numpy`, `pyyaml`. M3 adds
`librosa`, `scipy`. DSP stage A (M4) adds `pedalboard`. No torch at any point in this
roadmap; the RAVE/neural stage is a deferred optional addon, not scheduled here.

## Milestone 5: score / long-form (Eno-style notation)

Everything up to here is **serial**: `splice` lays segments end to end, and `tape_loop`
concatenates cycle 0, cycle 1, cycle 2 into one continuous file. That gives one evolving
loop. Long-form generative music (Eno's *Music for Airports*) needs the one primitive the
pipeline is missing: **parallel overlay on a timeline** — several loops of different
lengths running *at once*, summed into a master buffer. Because the loop lengths are
incommensurate (17.8s, 20.1s, 31.8s, ...) they never re-align, so a handful of short loops
generate 20+ minutes that never exactly repeat.

`tape_loop` is the single-voice, degenerate case of this. Generalize "one loop, serially
repeated, wearing per cycle" to "N loops, overlaid on a timeline, each wearing per cycle"
and the same `degrade` operator, seam joins, and seed discipline carry straight over.

### Mental model

A **score** is a set of **voices**. Each voice is a tape loop (render-once content that
`degrade`s as it repeats) placed on a shared timeline of total `duration`. Timing is
authored two ways, equal-weight, and both compile to the same flat schedule:

- **`period`** — the voice loops every `period` seconds from `offset`, filling `duration`.
  Pure Eno phasing; incommensurate periods across voices drive the evolution.
- **`events`** — explicit absolute-time triggers (a sparse hand-written score). A voice may
  carry both: a steady loop plus one-off punctuations.

Content per voice comes from either a **sample file** or the existing **collage chain**
(`slice -> sequence -> splice`) run on an input, rendered exactly once. Events may
**transpose** their source; transposition is **varispeed** (resample) by default — a tape
played faster is both higher and shorter, which is physically what pitch means on tape and
which feeds extra phasing back into `period` for free.

### Config (top-level `score` block)

```yaml
score:
  duration: 600            # total length in seconds — the long-form span
  normalize: -1.0          # dBFS ceiling; sum all voices then limit to this
  pitch_mode: varispeed    # varispeed (resample; default, tape-true) | timed (phase-vocoder)
  voices:
    - name: bell
      source: samples/bell_c.wav         # a sample file ...
      period: 17.8                        # ... looping every 17.8s (phasing)
      offset: 0.0                         # first entry time in seconds
      gain: 0.8
      pan: -0.3                           # -1..1, equal-power
      pitch: 0                            # semitones (float), voice default
      evolve: 0.15                        # tape wear accrued per cycle -> degrade()
      seam: crossfade                     # loop-point join: cut | zerocross | crossfade

    - name: drone
      source: { chain: inputs/pad.wav }   # ... or content GENERATED by the chain,
      period: 31.8                        #     rendered once, then looped/degraded
      gain: 0.5

    - name: melody
      source: samples/voice.wav
      events:                             # explicit triggers, peers with `period`
        - { at: 12.0,  pitch: 0,  gain: 1.0 }
        - { at: 47.5,  pitch: 3,  gain: 0.6 }   # +3 semitones from the one sample
        - { at: 190.0, pitch: -5, gain: 0.9 }
```

Per-voice keys default like the dials elsewhere: `offset: 0`, `gain: 1.0`, `pan: 0`,
`pitch: 0`, `evolve: 0`, `seam: crossfade`. `period` and `events` are both optional but a
voice needs at least one. Unknown keys are an error (fail loud), consistent with the loader
contract.

### Canonical IR: `Placement`

`period` + `events` for every voice compile down to one frozen object per triggered entry.
Freeze this first, the way `Segment` was frozen before M1.

```python
@dataclass(frozen=True)
class Placement:
    voice: str
    content: Path        # the render-once loop content for this voice
    start_frame: int     # absolute position on the master timeline
    cycle: int           # loop count so far -> wear = evolve * f(cycle)
    gain: float
    pan: float           # -1..1, equal-power
    pitch: float         # semitones; sign convention matches `degrade` wear ramp
```

The whole score is `list[Placement]` + `duration`. That list is what the sidecar records
and is hand-editable: nudge one `start_frame`, change one `pitch`, and re-render exactly.
`cycle` is the multi-voice heir to `Segment.cycle` — it drives `degrade(content,
evolve * cycle / span)` so each voice audibly ages as the piece unfolds.

### Determinism: per-voice sub-seeds

Derive each voice's RNG from `hash(seed, voice.name)`, not the shared `ctx.rng`, so voices
are independent: adding or removing a voice leaves every *other* voice's dropouts and wear
bit-identical. This is the multi-voice form of the tape_loop invariant ("bump cycles 8->9,
cycles 0-7 unchanged") and it makes iterating on a score sane. The master seed still governs
the whole run.

### Flow

1. **Render each voice's content once.** Sample source = load it; `{chain: <input>}` source
   = run the existing `slice -> sequence -> splice` chain to produce one `loop.wav`. This is
   the render-once win from tape_loop, now per voice.
2. **Compile the schedule.** Expand each voice's `period` into events at
   `offset, offset + period, ...` up to `duration`, merge with any explicit `events`, sort by
   time, assign each a `cycle` index. -> `list[Placement]`.
3. **Render each placement.** `degrade` the content by that cycle's wear; transpose by
   `pitch` (varispeed resample, or phase-vocoder under `pitch_mode: timed`).
4. **Mix onto the master timeline.** Allocate a `duration`-length buffer, sum each placement
   at its `start_frame` with equal-power pan, then limit to `normalize` (record the applied
   makeup/limiter gain). Stream out block by block via `io.BlockWriter` — never hold the
   whole piece in RAM.
5. **Sidecar** records `duration`, the resolved config, and the full compiled `list[Placement]`
   so a 10-minute generative piece re-renders bit-identically from the seed.

### Modules

New, mirroring the repo's grain; nothing existing is rewritten:
- `src/audiopipe/score.py` — parse the `score` block, render voice content, compile
  `period`/`events` -> `list[Placement]`, orchestrate.
- `src/audiopipe/mix.py` — the one genuinely new primitive: timeline allocation, equal-power
  pan, summing at absolute frame offsets, limiting, block-streamed output.
- `src/audiopipe/cli.py` — `audiopipe score config/score.yaml -o out.wav`. A score has many
  sources, so it is its own entrypoint, not the one-file `process`/`run` path.

Reused untouched: `degrade`, `splice` joins, `io.BlockWriter`, `sidecar`, the seeded
`Context`. `tape_loop` stays as the single-voice serial case; it is not folded into `score`.

### Build order

1. Freeze `Placement` + the compile step (`period`/`events` -> placements). Test the phasing
   math: incommensurate periods generate the right event times, cycle indexing, offsets.
2. `mix.py` timeline summing + equal-power pan + limit. Test: two silent-then-click voices
   land at the exact expected frames; output length == `duration`; sum never exceeds the
   `normalize` ceiling.
3. Wire pitch (varispeed) + per-voice `degrade`. Test determinism and sub-seed independence
   (remove a voice -> others bit-identical).
4. `{chain: <input>}` voice sources + CLI + sidecar of the full schedule.

Tests:
- A voice with `period: P` and no `events` over `duration: D` yields `floor((D - offset)/P)+1`
  placements at the expected frames.
- `period` + `events` on one voice merge and sort correctly; cycle indices are monotonic.
- Two voices with incommensurate periods do not re-align within `duration` (assert the
  combined onset pattern has no period <= duration).
- Mix of a click at t and a click at t' lands energy at exactly those frames; equal-power pan
  preserves total power; the limiter holds the ceiling with no sample past `normalize`.
- Varispeed pitch of +12 semitones halves a placement's length and doubles its rate; per-voice
  sub-seeds make voice removal a no-op for the others.
- Full score with a `{chain: ...}` voice renders once per voice (assert chain invoked once per
  voice regardless of cycle count), reproduces from the seed, and the sidecar round-trips the
  whole `list[Placement]`.

Dependencies added at M5: none beyond M3 (`scipy` already present for `degrade` and for the
varispeed resample; `pitch_mode: timed` reuses `librosa`). Still no torch.
