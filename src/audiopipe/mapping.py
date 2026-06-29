from __future__ import annotations

# Coarse 0..1 dials -> concrete values. Curves live here, out of the stages.

_LONG_S = 2.0    # grain seconds at amount 0 (long slices)
_SHORT_S = 0.05  # grain seconds at amount 1 (chopped)


def grain_frames(amount: float, sr: int) -> int:
    """amount 0 -> long grains, 1 -> short grains."""
    secs = _LONG_S * (1 - amount) + _SHORT_S * amount
    return max(1, int(secs * sr))


def dsp_params(cfg: dict) -> dict:
    """Expand coarse dsp dials (0..1) into concrete plugin params. A value of 0
    means the effect is omitted from the board. Curves live here, not the stage."""
    p = {}
    if cfg["drive"] > 0:
        p["drive_db"] = 30.0 * cfg["drive"]                 # 0..30 dB distortion
    if cfg["filter"] > 0:
        p["cutoff_hz"] = 18000 * (1 - cfg["filter"]) + 600 * cfg["filter"]  # high dial = darker
    if cfg["chorus"] > 0:
        p["chorus_mix"] = cfg["chorus"]                     # 0..1 wet
    if cfg["reverb"] > 0:
        p["reverb_room"] = cfg["reverb"]
        p["reverb_wet"] = 0.4 * cfg["reverb"]
    return p


# feel: sort -> librosa/feature key resolved by analyze.feature()
FEATURE_KEYS = {"brightness", "loudness", "duration"}


def feature_key(sort_by: str) -> str:
    if sort_by not in FEATURE_KEYS:
        raise ValueError(f"unknown sort_by {sort_by!r}; pick one of {sorted(FEATURE_KEYS)}")
    return sort_by
