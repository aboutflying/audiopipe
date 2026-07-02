from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import soundfile as sf
import yaml

from . import io, mix

# Per-voice defaults — transparent where a dial means "off".
_VOICE_DEFAULTS = {"offset": 0.0, "gain": 1.0, "pan": 0.0, "pitch": 0.0,
                   "wear": 0.0, "seam": "crossfade"}


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


def _with_defaults(voice: dict) -> dict:
    return {**_VOICE_DEFAULTS, **voice}


def render_score(config_path: Path, out_path: Path, sr: int = 44100) -> list[Placement]:
    """Prototype (M5 steps 1-2): sample-source voices only, overlaid on a timeline.
    Pitch, per-cycle wear, and {chain: ...} sources land in steps 3-4. The seed
    will drive per-voice sub-RNGs (hash(seed, voice.name)) once wear lands."""
    from . import sidecar
    cfg = yaml.safe_load(Path(config_path).read_text())["score"]
    duration = float(cfg["duration"])
    normalize = float(cfg.get("normalize", -1.0))
    seed = int(cfg.get("seed", 42))
    voices = [_with_defaults(v) for v in cfg["voices"]]

    content = {}
    for v in voices:
        src = v["source"]
        if isinstance(src, dict):
            raise NotImplementedError("{chain: ...} voice sources land in M5 step 4")
        buf, csr = sf.read(str(src), dtype="float32", always_2d=True)
        content[v["name"]] = io.resample(buf, csr, sr) if csr != sr else buf
        v["content"] = src

    placements = compile_placements(voices, duration, sr)
    master = mix.mix_placements(placements, content, int(round(duration * sr)),
                                normalize, sr=sr)
    sf.write(str(out_path), master, sr)
    sidecar.write_score(out_path, config=cfg, placements=placements, seed=seed)
    return placements
