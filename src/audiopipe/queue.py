from __future__ import annotations
from pathlib import Path


class Queue:
    """Directory state machine. A file moves inbox -> working -> done/failed.
    Debuggable by `ls`, survives restarts. Claim is an atomic rename."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.inbox = self.root / "inbox"
        self.working = self.root / "working"
        self.done = self.root / "done"
        self.failed = self.root / "failed"
        self.outbox = self.root / "outbox"
        for d in (self.inbox, self.working, self.done, self.failed, self.outbox):
            d.mkdir(parents=True, exist_ok=True)

    def list_new(self) -> list[Path]:
        return sorted(p for p in self.inbox.iterdir() if p.is_file())

    def claim(self, path: Path) -> Path | None:
        """Atomic move inbox -> working. Returns new path, or None if lost the
        race / already claimed. rename is atomic on a single filesystem."""
        dest = self.working / path.name
        try:
            path.rename(dest)
        except (FileNotFoundError, OSError):
            return None
        return dest

    def finish(self, working_path: Path) -> Path:
        dest = self.done / working_path.name
        working_path.rename(dest)
        return dest

    def fail(self, working_path: Path) -> Path:
        dest = self.failed / working_path.name
        working_path.rename(dest)
        return dest
