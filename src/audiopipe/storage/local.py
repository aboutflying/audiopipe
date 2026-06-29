from __future__ import annotations
import shutil
from pathlib import Path


class LocalStorage:
    """Local filesystem backend. fetch is a no-op (already local)."""

    def __init__(self, inbox: Path, outbox: Path):
        self.inbox = Path(inbox)
        self.outbox = Path(outbox)

    def list_new(self) -> list[Path]:
        return sorted(p for p in self.inbox.iterdir() if p.is_file())

    def fetch(self, path: Path) -> Path:
        return path

    def publish(self, local_path: Path) -> None:
        self.outbox.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, self.outbox / Path(local_path).name)
