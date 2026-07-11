"""
simgraph_adapter.py — DRVEngine-compatible wrapper around SimGraph.

Exposes the same attributes and methods as DRVEngine so server.py
can use it as a drop-in replacement.

Public surface (matches DRVEngine):
    .n, .m          shape
    .t              simulation time
    .P [n,m]        allocation fractions (writable, synced to nodes each tick)
    .W [m]          resistance          (writable, synced to nodes each tick)
    .G [m]          goals               (writable, synced to nodes each tick)
    .A [m]          accumulated progress (read from nodes after tick)
    .PHI [m,m]      gating matrix       (writable; edges rebuilt on change)
    .ALPHA [m,m]    boost matrix        (writable; edges rebuilt on change)
    .X [n,m]        eligibility mask    (read-only after init)

    .tick()                  → dict (same format as DRVEngine.tick)
    .contribution() → [n,m]
    .readiness()    → [m]
    .boost()        → [m]
"""

import numpy as np
from data import build_tensors
from build_simgraph import build_from_tensors


class SimGraphAdapter:
    """
    Wraps SimGraph to match the DRVEngine public interface.

    Strategy:
      - Hold numpy arrays (W, G, PHI, ALPHA, P) as the authoritative source.
      - Each tick(): push changed params into SimGraph nodes, then tick, then
        pull results back out into numpy arrays.
      - readiness() and boost() are computed analytically from the project S
        values — same formulas as DRVEngine but reading from SimGraph state.
    """

    def __init__(self, P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys):
        self.n = P.shape[0]
        self.m = P.shape[1]

        # Numpy arrays — primary source of truth for params
        self.P     = P.copy().astype(float)
        self.CAP   = CAP.copy().astype(float)
        self.W     = W.copy().astype(float)
        self.X     = X.copy().astype(float)
        self.G     = G.copy().astype(float)
        self.PHI   = PHI.copy().astype(float)
        self.ALPHA = ALPHA.copy().astype(float)

        self._names     = list(names)
        self._proj_keys = list(proj_keys)

        # Build initial SimGraph
        self._sim, self._person_ids, _ = build_from_tensors(
            self.P, self.CAP, self.W, self.X,
            H, self.G, self.PHI, self.ALPHA,
            names, proj_keys,
        )

        # Internal state mirrors
        self.A  = np.zeros(self.m)          # accumulated progress per project
        self.t  = 0.0
        self._C = np.zeros((self.n, self.m))  # contribution matrix cache

        # Track whether PHI/ALPHA edges need a rebuild
        self._phi_dirty   = False
        self._alpha_dirty = False

        # Snapshots for PHI/ALPHA comparison
        self._last_PHI   = PHI.copy()
        self._last_ALPHA = ALPHA.copy()

    # ── param sync helpers ────────────────────────────────────────────────────

    def _push_params(self):
        """Sync numpy arrays → SimGraph node params before each tick."""
        sim = self._sim

        # Person params
        for i, pid in enumerate(self._person_ids):
            node = sim.nodes[pid]
            node.params["CAP"] = float(self.CAP[i])
            # Rebuild allocs dict from P[i]
            allocs = {}
            for j, key in enumerate(self._proj_keys):
                if self.X[i, j] > 0 and self.P[i, j] > 0.001:
                    allocs[key] = float(self.P[i, j] * 100)
            node.params["allocs"] = allocs

        # Project params
        for j, key in enumerate(self._proj_keys):
            node = sim.nodes[key]
            node.params["W"] = float(self.W[j])
            node.params["G"] = float(self.G[j])

        # PHI/ALPHA edges: rebuild if matrices changed
        phi_changed   = not np.allclose(self.PHI,   self._last_PHI,   atol=1e-6)
        alpha_changed = not np.allclose(self.ALPHA, self._last_ALPHA, atol=1e-6)
        if phi_changed or alpha_changed:
            self._rebuild_dep_edges()
            self._last_PHI   = self.PHI.copy()
            self._last_ALPHA = self.ALPHA.copy()

    def _rebuild_dep_edges(self):
        """Remove all PHI/ALPHA edges and re-add from current matrices."""
        sim = self._sim
        # Remove edges with PHI/ALPHA labels
        sim.edges = [e for e in sim.edges if not (
            e.label.startswith("PHI[") or e.label.startswith("ALPHA[")
        )]
        sim._dirty = True

        n, m = self.n, self.m
        keys = self._proj_keys

        for j in range(m):
            for k in range(m):
                phi = float(self.PHI[j, k])
                if phi > 1e-6:
                    phi_c = phi
                    sim.connect(
                        keys[k], "readiness",
                        keys[j], "gate",
                        transform=lambda s, φ=phi_c: max(float(s), 1e-6) ** φ,
                        label=f"PHI[{j},{k}]={phi:.2f}",
                    )

        X = self.X
        P = self.P
        import math as _math
        for j in range(m):
            for k in range(m):
                alpha_val = float(self.ALPHA[j, k])
                if alpha_val < 1e-6:
                    continue
                a_c = alpha_val
                for i in range(n):
                    if X[i, j] > 0 and P[i, j] > 0.001:
                        sim.connect(
                            keys[k], "readiness",
                            f"p_{i}", "eta_boost",
                            transform=lambda s, α=a_c: _math.log1p(α * max(float(s), 0.0)),
                            label=f"ALPHA[{j},{k}]={alpha_val:.2f}→p_{i}",
                        )

    # ── pull results from SimGraph ────────────────────────────────────────────

    def _pull_results(self):
        """Pull SimGraph output into numpy arrays after tick."""
        sim  = self._sim
        keys = self._proj_keys

        # Contribution matrix C[n,m]
        for i, pid in enumerate(self._person_ids):
            contrib = sim.nodes[pid].out.get("contribution", {})
            for j, key in enumerate(keys):
                self._C[i, j] = contrib.get(key, 0.0)

        # Accumulated progress A[m]
        for j, key in enumerate(keys):
            node = sim.nodes[key]
            self.A[j] = node.state.get("A", 0.0)

        self.t = self._sim.t

    # ── public interface ──────────────────────────────────────────────────────

    def tick(self, dt: float = 1.0) -> dict:
        self._push_params()
        self._sim.tick(dt)      # advance SimGraph — its return format differs from DRVEngine
        self._pull_results()    # pulls _C[n,m], A[m], t back into numpy

        # Build DRVEngine-compatible dict so sim_loop can read state['S'] etc.
        flow = self._C.sum(axis=0).astype(np.float32)
        V    = self._C.sum(axis=1).astype(np.float32)
        S    = np.array([
            self._sim.nodes[k].state.get("S", 0.0)
            for k in self._proj_keys
        ], dtype=np.float32)
        R    = np.array([
            self._sim.nodes[k].state.get("R", 1.0)
            for k in self._proj_keys
        ], dtype=np.float32)
        return {
            "C":    self._C.copy().astype(np.float32),
            "flow": flow,
            "R":    R,
            "B":    self.boost().astype(np.float32),
            "A":    self.A.astype(np.float32),
            "V":    V,
            "S":    S,
            "t":    self.t,
        }

    def contribution(self) -> np.ndarray:
        """Return C[n,m] from last tick."""
        return self._C.copy()

    def readiness(self) -> np.ndarray:
        """
        R[j] = product_k  S[k]^PHI[j,k]   (log domain for numerical safety)
        """
        S = np.array([
            self._sim.nodes[k].state.get("S", 0.0)
            for k in self._proj_keys
        ], dtype=float)
        log_S = np.log(np.clip(S, 1e-6, 1.0))
        log_R = self.PHI @ log_S
        return np.exp(log_R)

    def boost(self) -> np.ndarray:
        """
        B[j] = product_k  (1 + ALPHA[j,k] * S[k])
        """
        S = np.array([
            self._sim.nodes[k].state.get("S", 0.0)
            for k in self._proj_keys
        ], dtype=float)
        log_B = np.sum(np.log1p(self.ALPHA * S[np.newaxis, :]), axis=1)
        return np.exp(log_B)

    def get_S(self) -> np.ndarray:
        return np.array([
            self._sim.nodes[k].state.get("S", 0.0)
            for k in self._proj_keys
        ], dtype=float)

    # ── SimGraph-specific extras (not in DRVEngine) ───────────────────────────

    @property
    def sim(self) -> object:
        """Direct access to underlying SimGraph for inspection or extension."""
        return self._sim

    def add_node(self, node) -> None:
        self._sim.add(node)

    def remove_node(self, node_id: str) -> None:
        self._sim.remove(node_id)
        # If it was a person, remove from person_ids
        if node_id in self._person_ids:
            idx = self._person_ids.index(node_id)
            self._person_ids.pop(idx)
            self.n -= 1
            self._C = np.delete(self._C, idx, axis=0)
            self.CAP = np.delete(self.CAP, idx)
            self.P   = np.delete(self.P, idx, axis=0)
            self.X   = np.delete(self.X, idx, axis=0)

    def connect(self, *args, **kwargs):
        self._sim.connect(*args, **kwargs)

    def snapshot(self) -> dict:
        return {
            "t":    self.t,
            "A":    self.A.tolist(),
            "S":    self.get_S().tolist(),
            "R":    self.readiness().tolist(),
            "B":    self.boost().tolist(),
            "C":    self._C.tolist(),
        }


# ── Drop-in builder matching DRVEngine constructor signature ──────────────────

def build_adapter(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys) -> SimGraphAdapter:
    return SimGraphAdapter(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)
