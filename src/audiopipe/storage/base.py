from __future__ import annotations
from typing import Protocol
from pathlib import Path


class StorageBackend(Protocol):
    def list_new(self) -> list[Path]: ...
    def fetch(self, path: Path) -> Path: ...   # ensure local & readable; iCloud
                                               # impl will brctl-download here
    def publish(self, local_path: Path) -> None: ...
