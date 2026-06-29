from __future__ import annotations
from ..segment import EDL
from .base import Context


class Passthrough:
    name = "passthrough"

    def process(self, edl: EDL, ctx: Context) -> EDL:
        edl.record(self.name, {})
        return edl
