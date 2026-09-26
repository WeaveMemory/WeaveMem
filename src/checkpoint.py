

from __future__ import annotations

import time
import threading

import jsonio


class Checkpoint:
    """Tracks completed (group, unit) pairs in one JSON file; writes atomically."""

    def __init__(self, path: str, flow: str) -> None:
        self.path = path
        self.flow = flow
        self._lock = threading.RLock()
        data = jsonio.read_json(path, default=None) or {}
        raw = data.get("completed", {})
        # Store units as ints in sets for O(1) membership and clean de-duplication.
        self._completed: dict[str, set[int]] = {
            group: set(units) for group, units in raw.items()
        }

    def done(self, group: str, unit: int) -> bool:
        """True if this unit has already been completed for the group."""
        with self._lock:
            return unit in self._completed.get(group, set())

    def done_units(self, group: str) -> set[int]:
        """Set of completed units for the group (empty if none)."""
        with self._lock:
            return set(self._completed.get(group, set()))

    def mark(self, group: str, unit: int) -> None:
        """Record a unit as completed and persist immediately (atomic write)."""
        with self._lock:
            self._completed.setdefault(group, set()).add(unit)
            self._save()

    def clear(self, group: str | None = None) -> None:
        """Drop progress for one group, or all groups when group is None; persist."""
        with self._lock:
            if group is None:
                self._completed.clear()
            else:
                self._completed.pop(group, None)
            self._save()

    def _save(self) -> None:
        payload = {
            "flow": self.flow,
            "completed": {g: sorted(u) for g, u in self._completed.items()},
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        jsonio.atomic_write_json(self.path, payload)
