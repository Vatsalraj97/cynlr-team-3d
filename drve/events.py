"""
events.py — Discrete event queue for DRVE simulation.

Event types:
  GOAL_SHOCK      — scale all project goals by factor
  EFFICIENCY_LEAK — reduce H[i,j] by factor (simulates person burnout / mis-fit)
  PERSON_FREEZE   — zero out person i's allocations temporarily
  PERSON_THAW     — restore frozen person's previous allocations
  PRIORITY_SHIFT  — change W[j] for a project
  ALLOCATION_BUMP — force-set P[i,j] and re-normalise row
"""
import heapq
import numpy as np
from engine import DRVEngine


class EventQueue:
    def __init__(self):
        self._heap: list = []          # (fire_time, seq, event_type, payload)
        self._seq  = 0
        self._handlers: dict = {}
        self._fired: list = []         # log of all fired events
        self._frozen: dict = {}        # person_index → saved P row

        # register built-in handlers
        self.register("GOAL_SHOCK",      self._goal_shock)
        self.register("EFFICIENCY_LEAK", self._efficiency_leak)
        self.register("PERSON_FREEZE",   self._person_freeze)
        self.register("PERSON_THAW",     self._person_thaw)
        self.register("PRIORITY_SHIFT",  self._priority_shift)
        self.register("ALLOCATION_BUMP", self._allocation_bump)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, t: float, event_type: str, payload: dict | None = None):
        heapq.heappush(self._heap, (t, self._seq, event_type, payload or {}))
        self._seq += 1

    def register(self, event_type: str, handler):
        self._handlers.setdefault(event_type, []).append(handler)

    def process(self, engine: DRVEngine, current_t: float) -> list[tuple]:
        """Fire all events whose time ≤ current_t. Returns list of fired events."""
        fired = []
        while self._heap and self._heap[0][0] <= current_t:
            t, _, etype, payload = heapq.heappop(self._heap)
            for h in self._handlers.get(etype, []):
                h(engine, payload)
            record = (t, etype, payload)
            fired.append(record)
            self._fired.append(record)
        return fired

    def peek_next(self) -> float | None:
        """Return time of next scheduled event, or None."""
        return self._heap[0][0] if self._heap else None

    def log(self) -> list[tuple]:
        return list(self._fired)

    # ------------------------------------------------------------------
    # Built-in handlers
    # ------------------------------------------------------------------

    def _goal_shock(self, engine: DRVEngine, payload: dict):
        """Scale all project goals by payload['factor']."""
        engine.G *= float(payload.get("factor", 1.5))

    def _efficiency_leak(self, engine: DRVEngine, payload: dict):
        """Reduce H[i,j] by payload['factor'] (default 0.7)."""
        i = self._resolve_person(engine, payload)
        j = self._resolve_project(engine, payload)
        if i is not None and j is not None:
            engine.H[i, j] *= float(payload.get("factor", 0.7))
            engine.H[i, j]  = max(0.05, engine.H[i, j])
        elif i is not None:
            # Leak all of person i's projects
            engine.H[i] *= float(payload.get("factor", 0.7))
            engine.H[i]  = np.maximum(engine.H[i], 0.05)

    def _person_freeze(self, engine: DRVEngine, payload: dict):
        """Temporarily remove a person from all allocations."""
        i = self._resolve_person(engine, payload)
        if i is None:
            return
        self._frozen[i] = (engine.P[i].copy(), engine.X[i].copy())
        engine.P[i] = 0.0
        engine.X[i] = 0.0

    def _person_thaw(self, engine: DRVEngine, payload: dict):
        """Restore previously frozen person."""
        i = self._resolve_person(engine, payload)
        if i is None or i not in self._frozen:
            return
        engine.P[i], engine.X[i] = self._frozen.pop(i)

    def _priority_shift(self, engine: DRVEngine, payload: dict):
        """Change a project's strategic weight."""
        j = self._resolve_project(engine, payload)
        if j is not None:
            engine.W[j] = float(payload["new_weight"])

    def _allocation_bump(self, engine: DRVEngine, payload: dict):
        """Force-set P[i,j] = value and re-normalise row."""
        i = self._resolve_person(engine, payload)
        j = self._resolve_project(engine, payload)
        if i is None or j is None:
            return
        val = float(payload.get("value", 0.5))
        engine.P[i, j] = val
        engine.X[i, j] = 1.0
        # Re-normalise other allocations to fill remainder
        rest_mask = (engine.X[i] > 0) & (np.arange(engine.m) != j)
        rest_sum  = engine.P[i, rest_mask].sum()
        if rest_sum > 1e-6:
            engine.P[i, rest_mask] *= (1.0 - val) / rest_sum
        engine.P[i] = np.clip(engine.P[i], 0.0, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_person(engine: DRVEngine, payload: dict) -> int | None:
        if "person_idx" in payload:
            return int(payload["person_idx"])
        name = payload.get("person")
        if name and name in engine.names:
            return engine.names.index(name)
        return None

    @staticmethod
    def _resolve_project(engine: DRVEngine, payload: dict) -> int | None:
        if "project_idx" in payload:
            return int(payload["project_idx"])
        key = payload.get("project")
        if key and key in engine.proj_keys:
            return engine.proj_keys.index(key)
        return None
