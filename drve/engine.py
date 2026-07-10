"""
engine.py — DRVE forward pass (v2: resistance + dependency model).

State tensors (all float32 numpy):
  P   [n,m]  allocation fractions p_ij, rows sum to 1
  H   [n,m]  base efficiency η_ij (default 1.0; modified by events)
  CAP [n]    capability scores
  W   [m]    resistance — difficulty of project j  (higher = harder)
  X   [n,m]  eligibility mask
  A   [m]    accumulated progress
  G   [m]    project goals (total energy to complete)
  PHI [m,m]  progress coupling — PHI[j,k]: k gates j's progress
  ALPHA[m,m] efficiency coupling — ALPHA[j,k]: k boosts η on j

One tick:
  C[i,j]     = CAP_i * p_ij * η_ij(t)           raw contribution
  η_ij(t)    = H[i,j] * Π_k (1 + ALPHA[j,k]*S[k])  tool-boosted
  FLOW[j]    = Σ_i C[i,j]
  R[j]       = Π_k  clip(S[k],0,1)^PHI[j,k]    readiness
  dA[j]/dt   = (FLOW[j] / W[j]) * R[j]
"""
import numpy as np
from typing import Optional


class DRVEngine:
    def __init__(
        self,
        P:     np.ndarray,            # [n,m]
        CAP:   np.ndarray,            # [n]
        W:     np.ndarray,            # [m]  resistance
        X:     np.ndarray,            # [n,m]
        H:     Optional[np.ndarray] = None,   # [n,m] base efficiency
        G:     Optional[np.ndarray] = None,   # [m]   goals
        PHI:   Optional[np.ndarray] = None,   # [m,m] progress coupling
        ALPHA: Optional[np.ndarray] = None,   # [m,m] efficiency coupling
        names:      Optional[list] = None,
        proj_keys:  Optional[list] = None,
    ):
        self.n, self.m = P.shape
        self.P     = P.astype(np.float32).copy()
        self.CAP   = CAP.astype(np.float32).copy()
        self.W     = W.astype(np.float32).copy()
        self.X     = X.astype(np.float32).copy()
        self.H     = H.astype(np.float32).copy()     if H     is not None else np.ones((self.n, self.m), np.float32)
        self.G     = G.astype(np.float32).copy()     if G     is not None else np.ones(self.m, np.float32) * 1000.0
        self.PHI   = PHI.astype(np.float32).copy()   if PHI   is not None else np.zeros((self.m, self.m), np.float32)
        self.ALPHA = ALPHA.astype(np.float32).copy() if ALPHA is not None else np.zeros((self.m, self.m), np.float32)
        self.A     = np.zeros(self.m, dtype=np.float32)
        self.t     = 0.0
        self.names     = names     or [str(i) for i in range(self.n)]
        self.proj_keys = proj_keys or [str(j) for j in range(self.m)]
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Core math
    # ------------------------------------------------------------------

    def readiness(self) -> np.ndarray:
        """R[j] = Π_k clip(S[k],0,1)^PHI[j,k]  — dependency gates."""
        S = np.clip(self.A / np.maximum(self.G, 1e-6), 0.0, 1.0)
        # R[j] = product over k of S[k]^PHI[j,k]
        # log domain: log R[j] = Σ_k PHI[j,k] * log(clip(S[k], ε, 1))
        log_S = np.log(np.maximum(S, 1e-6))
        log_R = self.PHI @ log_S          # [m]
        return np.exp(log_R)              # [m], in (0,1]

    def boost(self) -> np.ndarray:
        """B[j] = Π_k (1 + ALPHA[j,k]*S[k]) — tool efficiency multiplier."""
        S = np.clip(self.A / np.maximum(self.G, 1e-6), 0.0, 1.0)
        # log domain: log B[j] = Σ_k log(1 + ALPHA[j,k]*S[k])
        log_B = np.sum(np.log1p(self.ALPHA * S[None, :]), axis=1)   # [m]
        return np.exp(log_B)

    def effective_eta(self) -> np.ndarray:
        """eta_ij = H[i,j] * B[j]  (broadcast)  — [n,m]."""
        return self.H * self.boost()[None, :]

    def contribution(self) -> np.ndarray:
        """C[n,m] = CAP_i * p_ij * eta_ij(t)  (W removed — now in progress eq)."""
        return self.CAP[:, None] * self.P * self.effective_eta()

    def tick(self, dt: float = 1.0) -> dict:
        C    = self.contribution()                          # [n,m]
        flow = C.sum(axis=0)                               # [m]
        R    = self.readiness()                            # [m]
        W    = np.maximum(self.W, 1e-3)
        # dA/dt = (FLOW / W) * R
        self.A += (flow / W) * R * dt
        V = C.sum(axis=1)                                  # [n]
        S = np.clip(self.A / np.maximum(self.G, 1e-6), 0.0, 2.0)

        self.t += dt
        state = {
            "C":    C.copy(),
            "flow": flow.copy(),
            "R":    R.copy(),
            "B":    self.boost().copy(),
            "A":    self.A.copy(),
            "V":    V.copy(),
            "S":    S.copy(),
            "t":    self.t,
        }
        self._history.append(state)
        return state

    def reset_accumulator(self):
        self.A[:] = 0.0
        self.t    = 0.0
        self._history.clear()

    # ------------------------------------------------------------------
    # Convenience snapshots
    # ------------------------------------------------------------------

    def util(self) -> np.ndarray:
        """Strategic utilisation ratio V_i / CAP_i  [n]"""
        C = self.contribution()
        V = C.sum(axis=1)
        return V / np.maximum(self.CAP, 1e-6)

    def project_deficit(self) -> np.ndarray:
        """gap_j = max(0, G_j - A_j)  [m]"""
        return np.maximum(0.0, self.G - self.A)

    def person_index(self, name: str) -> int:
        return self.names.index(name)

    def project_index(self, key: str) -> int:
        return self.proj_keys.index(key)

    # ------------------------------------------------------------------
    # Serialise / clone
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in self.__dict__.items()
                if k != "_history"}

    def restore(self, snap: dict):
        for k, v in snap.items():
            setattr(self, k, v.copy() if isinstance(v, np.ndarray) else v)
        self._history.clear()
