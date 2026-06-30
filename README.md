# audiopipe

A self-hosted audio pipeline that watches a folder, slices / resequences /
processes audio tape-collage style, and writes the result back with a
reproducible metadata sidecar.

The core idea: every stage is `EDL -> EDL`. An **EDL** (edit decision list) is an
ordered list of `Segment`s — slices of audio held *by reference* (source +
frame bounds), never loaded into memory until rendered. A continuous file is
just an EDL with one segment, so there is no separate "continuous mode." The
chain order is declared in config, so `grain -> fx -> rearrange` vs
`rearrange -> grain -> fx` is a config reorder, never a code change.

Everything is driven by a single `seed`, so any output reproduces exactly —
a happy accident can be re-rendered, then tweaked one dial at a time.

## Install

Requires Python 3.10+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[analysis,fx]"
```

Dependency tiers (install only what you use):

| Extra | Pulls in | Needed for |
|-------|----------|------------|
| *(core)* | `soundfile`, `numpy`, `pyyaml` | grain / rearrange / splice |
| `analysis` | `librosa`, `scipy` | `onset`/`silence` slicing, `feel: sort`, tape-loop degrade |
| `fx` | `pedalboard` | the `fx` stage |

**M4A / AAC input** (e.g. `.m4a`) isn't decodable by libsndfile. On macOS,
audiopipe transcodes it automatically via the built-in `afconvert`. On other
platforms, transcode to WAV first.

## Quickstart

audiopipe processes files through a **work directory** — a state machine you
can inspect with `ls`:

```
work/
  inbox/      # drop new files here
  working/    # claimed, mid-render
  done/       # finished originals
  failed/     # errored (original preserved + error sidecar)
  outbox/     # rendered output + .json sidecar
```

Drain the inbox with the built-in default chain (`grain -> rearrange -> splice`):

```bash
mkdir -p work/inbox
cp mytrack.wav work/inbox/
.venv/bin/audiopipe process
# -> work/outbox/mytrack.wav  (+ mytrack.json)
#    original moved to work/done/
```

Run a single file directly, choosing a preset and output path:

```bash
.venv/bin/audiopipe -c config/presets/underwater.yaml run mytrack.wav -o out.wav
```

### CLI

```
audiopipe [-c CONFIG] [-w WORK] process       # drain work/inbox/
audiopipe [-c CONFIG] [-w WORK] run FILE [-o OUT]
```

- `-c, --config` — pipeline YAML. **Omit to use the built-in defaults.** Pass a
  file from `config/presets/` to override.
- `-w, --work` — work directory root (default `work/`).
- `run -o` — output path (default `work/outbox/<name>.wav`).

A file's processing is idempotent: same input + config + seed → same output.
Re-running a failed file is safe.

## Presets

`config/presets/` holds ready-made configs you pass with `-c`:

| Preset | Sound |
|--------|-------|
| `clean.yaml` | gentle grains, light tone/reverb, long crossfades |
| `blown.yaml` | heavy distortion, drifted grains, hard-cut splices |
| `underwater.yaml` | dark lowpass + chorus + big reverb, smeared joins |
| `tape.yaml` | a short loop disintegrating over 6 tape cycles |

Copy one and edit to taste — there is no default config file; the defaults
below live in code and apply whenever a key is omitted.

## Configuration reference

A config is YAML. **Unknown keys are an error** (fail loud); any omitted key
takes its default. Dials are coarse, roughly `0..1` where noted — the loader
and `mapping.py` expand them into concrete values.

```yaml
seed: 42                   # global determinism; change to roll a new variant
```

### `source`

```yaml
source:
  channels: sum            # sum | left | keep
  sample_rate: source      # "source" (keep input rate) or an int e.g. 44100
```

- `channels` — how to fold channels: `sum` (mix to mono), `left` (take left
  channel), `keep` (preserve all channels, e.g. stereo through the whole chain).
- `sample_rate` — `source` keeps the input rate; an integer would resample
  (the resample seam exists; not exercised by the default stages yet).

### `chain`

```yaml
chain:
  - grain
  - rearrange
  - splice
```

**Order is the composition.** Each name must be a registered stage; a stage
absent from the list is skipped. Available: `grain`, `rearrange`, `splice`,
`fx`, `vari`, `passthrough`. Reordering needs no code change. Typical: put `fx`
anywhere (`grain -> fx -> rearrange -> splice` effects each grain;
`... -> splice -> fx`… isn't possible since splice ends the chain — `fx`
before `splice` to colour grains, after `grain` to colour the raw cuts).

### `grain` — cut one segment into many

```yaml
grain:
  mode: grid               # grid | random | onset | silence
  density: 0.6             # low = long grains, high = chopped
  drift: 0.3               # 0 = clean grid; higher randomizes boundaries
```

- `mode`:
  - `grid` — fixed-length grains.
  - `random` — random lengths bounded by `density`.
  - `onset` — cut on detected transients *(needs `analysis`)*.
  - `silence` — cut on silence boundaries *(needs `analysis`)*.
- `density` — `0` → ~2 s grains, `1` → ~0.05 s grains (linear in between).
- `drift` — perturbs each boundary by up to `±drift × grain length`; stays
  within source bounds, never makes a negative-length grain.

### `rearrange` — reorder / thin the segments

```yaml
rearrange:
  feel: shuffle            # shuffle | reverse | as-is | sort
  scramble: 0.7            # shuffle: how far segments stray from order
  drop: 0.1                # fraction of segments discarded (0 = keep all)
  sort_by: brightness      # sort: brightness | loudness | duration
```

- `feel`:
  - `shuffle` — seeded reorder; `scramble` 0 keeps order, higher displaces more.
  - `reverse` — reverse the segment order.
  - `as-is` — keep order (use with `drop` to only thin).
  - `sort` — order by a feature *(needs `analysis`)*; also drops the **lowest**
    `drop` fraction (e.g. quietest) rather than random.
- `scramble` — shuffle displacement amount (`sort`/`reverse`/`as-is` ignore it).
- `drop` — fraction removed. For non-`sort` feels the removed segments are
  chosen by the seeded RNG.
- `sort_by` — feature key for `feel: sort`: `brightness` (spectral centroid),
  `loudness` (RMS), or `duration`.

### `splice` — render segments to one continuous file

```yaml
splice:
  join: crossfade          # cut | zerocross | crossfade
  fade: 0.2                # crossfade length (coarse)
```

This is where segments are materialized (windowed reads, block-by-block write).
- `join`:
  - `cut` — hard butt-join (clicky on purpose).
  - `zerocross` — snap cut points to zero crossings to reduce clicks.
  - `crossfade` — equal-power crossfade, length from `fade`.
- `fade` — crossfade length, ~5 ms at `0` up to ~100 ms at `1`.

### `fx` — pedalboard effects *(needs `fx` extra)*

```yaml
fx:
  drive: 0.2               # distortion, 0..30 dB
  tone: 0.3                # lowpass: high dial = darker
  chorus: 0.0              # modulation wet mix
  reverb: 0.25             # room size / wet
```

A sample-transforming stage: applies the effect chain to each segment and
writes rendered audio to scratch. Each dial is `0..1`; **`0` omits that
effect**. Only takes effect when `fx` is in `chain`.

The dials are coarse wrappers; each maps to one [pedalboard](https://spotify.github.io/pedalboard/)
plugin — see the [plugin reference](https://spotify.github.io/pedalboard/reference/pedalboard.html)
for what each underlying effect does:

| Dial | pedalboard plugin | Mapping (`mapping.fx_params`) |
|------|-------------------|--------------------------------|
| `drive` | [`Distortion`](https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.Distortion) | `drive_db` = `0..30` dB |
| `tone` | [`LowpassFilter`](https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.LowpassFilter) | `cutoff_frequency_hz` ≈ 18 kHz (`0`) → 600 Hz (`1`) |
| `chorus` | [`Chorus`](https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.Chorus) | `mix` = dial value |
| `reverb` | [`Reverb`](https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.Reverb) | `room_size` = dial, `wet_level` ≈ `0.4 × dial` |

To expose more pedalboard parameters or effects, extend `mapping.fx_params`
and `fx._build_board`.

### `vari` — reverse / varispeed per segment

```yaml
vari:
  reverse: 0.0             # probability (0..1) a grain plays backwards; 1 = all
  speed: 1.0              # playback rate: >1 faster+higher, <1 slower+lower
  wobble: 0.0        # per-grain random speed spread (tape wobble)
```

A sample-transforming stage (writes changed grains to scratch; untouched grains
stay reference-only). Only takes effect when `vari` is in `chain`.

- `reverse` — fraction of grains played backwards, chosen by the seeded RNG.
  `0` none, `1` all, `0.5` ≈ half. (This reverses the **audio**; to reverse the
  **order** of grains instead, use `rearrange: {feel: reverse}`.)
- `speed` — varispeed rate multiplier. `2.0` = an octave up at half length,
  `0.5` = an octave down at double length. Pitch follows speed, like a tape
  machine (it resamples; it does not pitch-preserve time-stretch).
- `wobble` — randomizes each grain's speed by up to `±wobble`, for
  drifting tape-wobble pitch.

### `tape_loop` — render-once, degrade-per-cycle *(needs `analysis` extra)*

```yaml
tape_loop:
  cycles: 1                # 1 = off (plain single render)
  wear: 0.4              # accumulated wear per cycle (0 = faithful repeats)
  feedback: false         # false = f(cycle) [cheap]; true = iterate the op
  seam: crossfade          # loop-point join: cut | zerocross | crossfade
  region: null             # null = whole output; [start_sec, end_sec] = a window
```

A post-chain construct (not a `chain` entry). The chain renders the loop
content **once**; then `cycles` copies are made and only a cheap *degrade*
operator (lowpass roll-off, level loss, dropouts) runs per cycle, ramped by
`wear × cycle / (cycles − 1)` — so cycle 0 is untouched and the last cycle is
fully worn.
- `cycles` — number of revolutions; `1` disables the loop entirely.
- `wear` — how much wear accumulates by the final cycle.
- `region` — which section of the rendered chain output to loop, in **seconds of
  the spliced output**. `null` loops the whole thing; `[8.0, 12.0]` loops only
  that 4-second window. Only that window is read, then cycled and degraded.
- `feedback` — `false` computes each cycle from the original (cheap, cycles
  independent); `true` degrades the previous cycle's output (for self-feeding
  effects like saturation).
- `seam` — the join at the loop point, kept separate from `splice.join` so the
  loop can click while internal cuts stay smooth.

## The sidecar

Every output gets a `<name>.json` next to it recording everything needed to
reproduce or hand-edit the render:

- `input` path and `input_sha256` (hash of the **original** source bytes)
- the full resolved `config` (including `seed`)
- the final `edl` — every segment's source, frame bounds, and op trail
- `audiopipe_version`, `timestamp`

On failure, the sidecar (in `work/failed/`) instead records the exception and
traceback, with the original input preserved.

## How it fits together

```
io.py        windowed audio read/write, channel policy, M4A transcode seam
segment.py   Segment + EDL — the reference-only intermediate representation
stages/      Stage protocol + Context (scratch dir, seeded RNG, channel policy)
segmenter / sequencer / splice / fx   the EDL -> EDL stages
analyze.py   librosa features (onset, silence, brightness, loudness)
mapping.py   coarse 0..1 dials -> concrete values (the curves live here)
degrade.py / tape_loop.py   the render-once tape construct
pipeline.py  config loader + validation + stage registry + runner
queue.py     the inbox/working/done/failed directory state machine
storage/     StorageBackend protocol (local implemented; S3/Dropbox later)
sidecar.py   the reproducibility record
cli.py       `audiopipe run` / `audiopipe process`
runner.py    orchestration: transcode -> chain -> tape_loop -> render -> sidecar
```

## Development

```bash
.venv/bin/python -m pytest        # full suite
```

Tests that need optional deps (`pedalboard`, macOS `afconvert`) skip
automatically when those aren't available.

### Status

- **M1** skeleton + passthrough — done
- **M2** collage core (grain / rearrange / splice) — done
- **M3** analysis (onset/silence/sort) + tape loop — done
- **M4** DSP stage — done · watcher daemon + remote storage backends — planned
