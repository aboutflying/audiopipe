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
| `analysis` | `librosa`, `scipy` | `onset`/`silence` slicing, `feel: sort`, tape-loop degrade, `ott` |
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
| `oldtape.yaml` | worn playback: saturation, dull highs, hiss, dropouts, wow & flutter |

Copy one and edit to taste — there is no default config file; the defaults
below live in code and apply whenever a key is omitted.

## Configuration reference

A config is YAML. **Unknown keys are an error** (fail loud); any omitted key
takes its default. Dials are coarse, roughly `0..1` where noted — the loader
and `mapping.py` expand them into concrete values.

**Defaults are transparent.** Every omitted dial is a no-op, so a config only
declares what it wants to change. The default chain is an *identity render*
(clean grid grains, kept in order, hard-cut back together = the input). You opt
into each effect by setting its dial; you never have to switch one off.

```yaml
seed: 42                   # global determinism; change to roll a new variant
```

### `source`

```yaml
source:
  channels: keep           # keep | sum | left   (keep = transparent)
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
`fx`, `vari`, `ott`, `passthrough`. Reordering needs no code change. Typical: put `fx`
anywhere (`grain -> fx -> rearrange -> splice` effects each grain;
`... -> splice -> fx`… isn't possible since splice ends the chain — `fx`
before `splice` to colour grains, after `grain` to colour the raw cuts).

### `grain` — cut one segment into many

```yaml
grain:
  mode: grid               # grid | random | onset | silence
  density: 0.6             # low = long grains, high = chopped
  drift: 0.0               # 0 = clean grid; higher randomizes boundaries
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
  feel: as-is              # as-is | shuffle | reverse | sort   (as-is = transparent)
  scramble: 0.0            # shuffle: how far segments stray from order
  drop: 0.0                # fraction of segments discarded (0 = keep all)
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
  join: cut                # cut | zerocross | crossfade   (cut = transparent)
  fade: 0.0                # crossfade length (coarse)
  dropouts: 0.0            # printed-in tape dropouts (baked into the render)
```

This is where segments are materialized (windowed reads, block-by-block write).
- `join`:
  - `cut` — hard butt-join (clicky on purpose).
  - `zerocross` — snap cut points to zero crossings to reduce clicks.
  - `crossfade` — equal-power crossfade, length from `fade`.
- `fade` — crossfade length, ~5 ms at `0` up to ~100 ms at `1`.
- `dropouts` — random short signal dropouts (~10-50 ms, declicked with a ~3 ms
  fade) **printed into the rendered output**. Because they're baked at render
  time, a `tape_loop` repeats the *same* holes each revolution, like physical
  tape damage. `0` = none; higher = more holes.

### `fx` — pedalboard effects *(needs `fx` extra)*

```yaml
fx:
  drive: 0.0               # distortion, 0..30 dB
  tone: 0.0                # lowpass: high dial = darker
  chorus: 0.0              # modulation wet mix
  reverb: 0.0              # room size / wet
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

### `ott` — extreme multiband compression *(needs `analysis` extra)*

```yaml
ott:
  depth: 0.0               # 0 = bypass; toward 1 = slammed wall-of-sound
  where: grain             # grain (per-grain, in the chain) | output (master pass)
```

OTT-style multiband upward **and** downward compression: the signal is split
into 3 bands, each band squashes what's loud *and* lifts what's quiet toward a
depth-scaled threshold, with strong makeup and a soft-clipped output. The result
is dense and loud (it raises RMS and crushes crest factor), and—being upward—it
drags up reverb tails, room tone, and tape hiss.

- `depth` — `0` bypasses; higher = lower threshold, higher ratios, more makeup.
- `where`:
  - `grain` — runs as a **chain stage**: each grain is slammed independently
    (pumpy, glitchy, on-brand for collage). List `ott` in `chain` where you want
    it, e.g. `chain: [grain, ott, splice]`.
  - `output` — runs as a **master pass** on the final rendered file, *after*
    `splice`/`tape_loop` (so it pumps the tape hiss too). Chain membership is
    ignored in this mode; just set `where: output`.

### `tape_loop` — finishing tape pass + render-once loop *(needs `analysis` extra)*

```yaml
tape_loop:
  cycles: 1                # 1 = single finishing pass; >1 = loop that disintegrates
  wear: 0.0                # ramped roll-off + level loss across cycles (loop only)
  feedback: false          # false = f(cycle) [cheap]; true = iterate the op
  seam: cut                # loop-point join: cut | zerocross | crossfade
  region: null             # null = whole output; [start_sec, end_sec] = a window
  hiss: 0.0                # tape noise floor (every pass, incl. cycles:1)
  flutter: 0.0             # wow & flutter (timebase warble) on the whole output
  speed: 1.0               # steady varispeed: >1 faster+higher, <1 slower+lower
```

A post-chain construct (not a `chain` entry) that applies **tape character** to
the rendered output. It runs whenever `cycles > 1` **or** any character dial
(`hiss`/`flutter`/`speed`) is set. (Dropouts live at `splice.dropouts` — they're
printed into the render so a loop repeats the same holes.)

Two roles:
- **Finishing pass** (`cycles: 1`): one tape pass over the spliced output —
  `speed` + `flutter` + `hiss`, no loop. This is the "old tape" use.
- **Disintegrating loop** (`cycles > 1`): the chain renders the loop content
  **once**, then `cycles` copies are made and a cheap *degrade* runs per cycle,
  with `wear` ramped by `wear × cycle / (cycles − 1)` — cycle 0 clean, the last
  fully worn. The character dials apply on every cycle on top of the wear ramp.

Dials:
- `cycles` — `1` = single finishing pass; `>1` = N revolutions that wear out.
- `wear` — roll-off + level loss accumulated by the final cycle (loop only;
  cycle 0 is always clean).
- `hiss` — additive tape noise floor (`1` ≈ −34 dB). Applied every pass.
- `flutter` — wow & flutter: a slow + fast LFO warps the timebase so pitch
  drifts (`vari.wobble` is the per-grain version; this is on the whole output).
- `speed` — steady varispeed on the whole output (`vari.speed` is the per-grain
  version). `2.0` = octave up / half length; pitch follows speed, like a tape.
- `region` — which section of the rendered output to use, in **seconds of the
  spliced output**. `null` = whole; `[8.0, 12.0]` = that 4-second window only.
- `feedback` — `false` computes each cycle from the original (cheap, cycles
  independent); `true` degrades the previous cycle's output (self-feeding
  saturation). Loop only.
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
