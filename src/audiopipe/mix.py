from __future__ import annotations
import numpy as np

# The one genuinely new primitive: parallel overlay on a shared timeline. Serial
# splice/tape_loop lay audio end-to-end; this sums voices *at once* at absolute
# frame offsets, which is what long-form (incommensurate loops) needs.


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power stereo pan, -1 (hard left) .. +1 (hard right). left^2+right^2 == 1."""
    theta = (float(pan) + 1) / 2 * (np.pi / 2)
    return float(np.cos(theta)), float(np.sin(theta))


def mix_placements(placements, content: dict, duration_frames: int,
                   normalize_db: float = -1.0) -> np.ndarray:
    """Allocate one stereo master buffer, sum each placement's content at its
    start_frame with equal-power pan and gain, then normalize the peak to the
    `normalize_db` ceiling. Returns the (frames, 2) master."""
    master = np.zeros((duration_frames, 2), dtype="float32")
    for p in placements:
        c = content[p.voice]
        if c.ndim == 2:
            c = c.mean(axis=1)                       # mono-ize for panning
        n = min(len(c), duration_frames - p.start_frame)
        if n <= 0:
            continue
        lg, rg = pan_gains(p.pan)
        seg = c[:n] * p.gain
        master[p.start_frame:p.start_frame + n, 0] += seg * lg
        master[p.start_frame:p.start_frame + n, 1] += seg * rg

    peak = float(np.max(np.abs(master)))
    if peak > 0:
        master *= (10 ** (normalize_db / 20)) / peak
    return master
