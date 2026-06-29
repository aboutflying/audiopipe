from __future__ import annotations
from .segment import EDL
from .stages.base import Context


class Sequencer:
    name = "sequence"

    def __init__(self, feel: str = "shuffle", strength: float = 0.7,
                 drop: float = 0.1, sort_by: str = "brightness"):
        self.feel = feel
        self.strength = float(strength)
        self.drop = float(drop)
        self.sort_by = sort_by  # used by feel: sort (M3)

    def process(self, edl: EDL, ctx: Context) -> EDL:
        segs = list(edl.segments)
        n = len(segs)
        k = round(self.drop * n)

        if self.feel == "sort":
            # feature-weighted: order by feature ascending, drop the lowest k
            # (e.g. quietest) fraction. Curve/key resolution via mapping.
            from .analyze import feature
            from .mapping import feature_key
            key = feature_key(self.sort_by)
            segs.sort(key=lambda s: feature(s, key))
            segs = segs[k:]
            segs = [s.with_op(f"seq:sort:{key}") for s in segs]
            edl.segments = segs
            edl.record(self.name, {"feel": "sort", "sort_by": self.sort_by,
                                   "drop": self.drop})
            return edl

        # non-sort feels: drop a random deterministic fraction first
        if k > 0:
            keep = set(range(n)) - set(ctx.rng.sample(range(n), k))
            segs = [s for i, s in enumerate(segs) if i in keep]

        if self.feel == "as-is":
            pass
        elif self.feel == "reverse":
            segs.reverse()
        elif self.feel == "shuffle":
            # key = index + strength*N*noise. strength 0 keeps order; higher
            # strength lets segments stray further from their slot.
            m = len(segs)
            jitter = [(i + self.strength * m * ctx.rng.uniform(-1, 1), s)
                      for i, s in enumerate(segs)]
            jitter.sort(key=lambda t: t[0])
            segs = [s for _, s in jitter]
        else:
            raise ValueError(f"unknown sequence feel {self.feel!r}")

        segs = [s.with_op(f"seq:{self.feel}") for s in segs]
        edl.segments = segs
        edl.record(self.name, {"feel": self.feel, "strength": self.strength,
                               "drop": self.drop})
        return edl
