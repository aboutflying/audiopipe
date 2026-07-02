from __future__ import annotations
from .segment import EDL
from .stages.base import Context


class Sequencer:
    name = "rearrange"

    def __init__(self, feel: str = "shuffle", scramble: float = 0.7,
                 drop: float = 0.1, sort_by: str = "brightness"):
        self.feel = feel
        self.scramble = float(scramble)
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
            segs = [s.with_op(f"rearrange:sort:{key}") for s in segs]
            edl.segments = segs
            edl.record(self.name, {"feel": "sort", "sort_by": self.sort_by,
                                   "drop": self.drop})
            return edl

        # non-sort feels: drop a random deterministic fraction first
        if k > 0:
            keep = set(range(n)) - set(ctx.rng_for(self.name).sample(range(n), k))
            segs = [s for i, s in enumerate(segs) if i in keep]

        if self.feel == "as-is":
            pass
        elif self.feel == "reverse":
            segs.reverse()
        elif self.feel == "shuffle":
            # key = index + scramble*N*noise. scramble 0 keeps order; higher
            # scramble lets segments stray further from their slot.
            m = len(segs)
            spread = [(i + self.scramble * m * ctx.rng_for(self.name).uniform(-1, 1), s)
                      for i, s in enumerate(segs)]
            spread.sort(key=lambda t: t[0])
            segs = [s for _, s in spread]
        else:
            raise ValueError(f"unknown rearrange feel {self.feel!r}")

        segs = [s.with_op(f"rearrange:{self.feel}") for s in segs]
        edl.segments = segs
        edl.record(self.name, {"feel": self.feel, "scramble": self.scramble,
                               "drop": self.drop})
        return edl
