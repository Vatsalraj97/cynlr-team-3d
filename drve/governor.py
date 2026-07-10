"""
governor.py — Gradient descent rebalancer with simplex projection.

Objective: minimise J = ½ Σ_j (V_target_j - Φ_j)²
Gradient:  ∂J/∂P_{i,j} = -(V_target_j - Φ_j) · CAP_i · W_j · H_{i,j}

After each gradient step the allocation matrix is:
  1. Clamped to [0, x_{i,j}]   — eligibility hard constraint
  2. Row-projected onto simplex — C1 constraint (Σ_j p_{i,j} = 1)

Simplex projection uses the Duchi et al. O(n log n) algorithm, which gives
exact projection (unlike softmax, which is only an approximation).
"""
import numpy as np
from engine import DRVEngine


# ------------------------------------------------------------------
# Simplex projection (Duchi et al. 2008)
# ------------------------------------------------------------------

def _simplex_project_vec(v: np.ndarray) -> np.ndarray:
    """Project a 1-D vector onto the probability simplex."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.where(u > (cssv - 1.0) / (np.arange(len(u)) + 1.0))[0]
    if len(rho_candidates) == 0:
        return np.zeros_like(v)
    rho = rho_candidates[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def simplex_project_masked(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Project v onto the simplex restricted to positions where mask==1.
    Ineligible positions are forced to zero.
    """
    eligible = mask.astype(bool)
    if not eligible.any():
        return np.zeros_like(v)
    result = np.zeros_like(v)
    result[eligible] = _simplex_project_vec(v[eligible])
    return result


def project_rows(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Apply masked simplex projection to every row of P."""
    P_out = np.empty_like(P)
    for i in range(P.shape[0]):
        P_out[i] = simplex_project_masked(P[i], X[i])
    return P_out


# ------------------------------------------------------------------
# Governor step
# ------------------------------------------------------------------

def governor_step(
    engine: DRVEngine,
    V_target: np.ndarray,   # [m] desired value flow per project per tick
    lr: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One gradient descent step toward V_target.

    Returns (flow_before, delta) for diagnostic use.
    """
    C        = engine.contribution()     # [n,m]
    flow     = C.sum(axis=0)            # [m] current Φ_j
    delta    = V_target - flow          # [m] error

    # Gradient of J w.r.t. P_{i,j}
    grad = -delta[None, :] * engine.CAP[:, None] * engine.W[None, :] * engine.H  # [n,m]

    engine.P -= lr * grad

    # Clamp negatives and mask ineligible slots
    engine.P = np.maximum(engine.P, 0.0)
    engine.P *= engine.X

    # Project each row back onto simplex (respecting eligibility)
    engine.P = project_rows(engine.P, engine.X)

    return flow, delta


# ------------------------------------------------------------------
# Exponentiated gradient (alternative — keeps positivity automatically)
# ------------------------------------------------------------------

def eg_step(
    engine: DRVEngine,
    V_target: np.ndarray,
    lr: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exponentiated gradient step: x_{n+1} ∝ x_n · exp(-η ∇J).
    Naturally keeps positivity; no clamping needed before normalisation.
    """
    C     = engine.contribution()
    flow  = C.sum(axis=0)
    delta = V_target - flow

    grad = -delta[None, :] * engine.CAP[:, None] * engine.W[None, :] * engine.H

    # Multiplicative update
    engine.P = engine.P * np.exp(-lr * grad)

    # Mask ineligible slots
    engine.P *= engine.X

    # Normalise each row (row-sum = 1)
    row_sums = engine.P.sum(axis=1, keepdims=True)
    engine.P = np.where(row_sums > 0, engine.P / row_sums, engine.P)

    return flow, delta


# ------------------------------------------------------------------
# LPP greedy solver (fast two-phase, same as webapp)
# ------------------------------------------------------------------

def lpp_solve(engine: DRVEngine) -> np.ndarray:
    """
    Greedy two-phase optimiser — maximises Σ C_{i,j}:
      Phase 1: sort person-project pairs by W_j descending, assign greedily
      Phase 2: remaining capacity routed to highest-W eligible project

    Returns optimal P* [n,m].
    """
    n, m = engine.n, engine.m
    P_opt = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        eligible = np.where(engine.X[i] > 0)[0]
        if len(eligible) == 0:
            continue
        # Sort by weight descending
        order = eligible[np.argsort(-engine.W[eligible])]
        # Assign 100% to highest-W eligible project
        P_opt[i, order[0]] = 1.0

    return P_opt
