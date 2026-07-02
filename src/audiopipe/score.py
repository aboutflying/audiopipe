from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
import yaml

from . import io, mix

# Per-voice defaults — transparent where a dial means "off".
_VOICE_DEFAULTS = {"offset": 0.0, "gain": 1.0, "pan": 0.0, "pitch": 0.0,
                   "wear": 0.0, "seam": "crossfade"}
_VOICE_KEYS = {"name", "source", "period", "events", *_VOICE_DEFAULTS}


@dataclass(frozen=True)
class Placement:
    """One triggered entry of a voice on the master timeline. `period` + `events`
    for every voice compile down to a flat list of these — the score's IR, the
    multi-voice heir to Segment. Hand-editable and re-renderable from the seed."""
    voice: str
    content: Path        # the render-once loop content for this voice
    start_frame: int     # absolute position on the master timeline
    cycle: int           # loop count so far; drives degrade wear = voice.wear * f(cycle)
    gain: float
    pan: float           # -1..1, equal-power
    pitch: float         # semitones (varispeed by default)


def _voice_entries(voice: dict, duration: float) -> list[tuple[float, float, float]]:
    """(time_s, gain, pitch) for one voice: its `period` grid plus any explicit
    `events`, merged and sorted by time. A voice needs at least one of the two."""
    entries = []
    offset = float(voice.get("offset", 0.0))
    if voice.get("period"):
        t, p = offset, float(voice["period"])
        while t < duration:
            entries.append((t, float(voice["gain"]), float(voice["pitch"])))
            t += p
    for ev in voice.get("events") or []:
        at = float(ev["at"])
        if at < duration:
            entries.append((at, float(ev.get("gain", voice["gain"])),
                            float(ev.get("pitch", voice["pitch"]))))
    if not entries:
        raise ValueError(f"voice {voice.get('name')!r} has neither period nor events")
    entries.sort(key=lambda e: e[0])
    return entries


def compile_placements(voices: list[dict], duration: float, sr: int) -> list[Placement]:
    """Expand each voice's period/events into a flat, time-sorted list of Placements,
    assigning a per-voice cycle index in time order."""
    out = []
    for v in voices:
        for cycle, (t, gain, pitch) in enumerate(_voice_entries(v, duration)):
            out.append(Placement(voice=v["name"], content=Path(v["content"]),
                                 start_frame=int(round(t * sr)), cycle=cycle,
                                 gain=gain, pan=float(v["pan"]), pitch=pitch))
    out.sort(key=lambda p: (p.start_frame, p.voice))
    return out


def prepare_clip(base: np.ndarray, sr: int, *, pitch: float, wear: float) -> np.ndarray:
    """One placement's audio: degrade by its cycle's wear, then transpose by
    varispeed — a tape played faster is higher AND shorter, so pitch feeds
    extra phasing back into the period structure for free."""
    from .degrade import degrade
    out = degrade(base, sr, wear)
    if pitch:
        factor = 2 ** (pitch / 12)
        out = io.resample_to(out, max(1, round(len(out) / factor)))
    return out


def _with_defaults(voice: dict) -> dict:
    unknown = set(voice) - _VOICE_KEYS
    if unknown:
        raise ValueError(f"unknown voice key(s) {sorted(unknown)} on "
                         f"{voice.get('name')!r}; allowed: {sorted(_VOICE_KEYS)}")
    return {**_VOICE_DEFAULTS, **voice}


def _render_voice_content(v: dict, sr: int, scratch: Path) -> np.ndarray:
    """A voice's loop content as a buffer, rendered exactly once. A plain path
    is a sample file; {chain: <input>, config: <pipeline.yaml>} runs the collage
    chain on the input (render-once: the expensive work never repeats per cycle)."""
    src = v["source"]
    if isinstance(src, dict):
        from .runner import render_one
        inp = Path(src["chain"])
        loop = scratch / f"voice_{v['name']}.wav"
        render_one(inp, src.get("config"), loop, scratch / f"{v['name']}_scratch")
        v["content"] = src["chain"]          # provenance points at the input
        buf, csr = sf.read(str(loop), dtype="float32", always_2d=True)
    else:
        v["content"] = src
        buf, csr = sf.read(str(src), dtype="float32", always_2d=True)
    return io.resample(buf, csr, sr) if csr != sr else buf


def render_score(config_path: Path, out_path: Path, sr: int = 44100) -> list[Placement]:
    """Render a score: N voices (sample or chain-generated loops) overlaid on one
    timeline, each placement degraded by its cycle's wear and transposed by its
    pitch, mixed with equal-power pan and limited. Sidecar records everything."""
    from . import sidecar
    cfg = yaml.safe_load(Path(config_path).read_text())["score"]
    duration = float(cfg["duration"])
    normalize = float(cfg.get("normalize", -1.0))
    seed = int(cfg.get("seed", 42))
    voices = [_with_defaults(v) for v in cfg["voices"]]

    with tempfile.TemporaryDirectory(prefix="audiopipe_score_") as tmp:
        content = {v["name"]: _render_voice_content(v, sr, Path(tmp)) for v in voices}

    placements = compile_placements(voices, duration, sr)
    wear_by_voice = {v["name"]: float(v["wear"]) for v in voices}
    # wear ramps 0 -> full across each voice's own cycle count (last cycle fully worn)
    spans = {}
    for p in placements:
        spans[p.voice] = max(spans.get(p.voice, 0), p.cycle)
    clips = [prepare_clip(content[p.voice], sr, pitch=p.pitch,
                          wear=wear_by_voice[p.voice] * p.cycle / max(spans[p.voice], 1))
             for p in placements]

    master = mix.mix_placements(placements, clips, int(round(duration * sr)),
                                normalize, sr=sr)
    sf.write(str(out_path), master, sr)
    sidecar.write_score(out_path, config=cfg, placements=placements, seed=seed)
    return placements
