"""
feedback.py — Reliability index and back-calculated efficiency.

R_i  = 1 - rolling_mean(|ΔV_i|) / (V_i + ε)     ∈ [0, 1]
η'_{i,j} = C_{i,j} / (CAP_i · p_{i,j} · W_j)    (back-calculated)

People with R_i < R_THRESHOLD are flagged as High Entropy.
"""
import numpy as np
from engine import DRVEngine

R_THRESHOLD = 0.70


def compute_reliability(engine: DRVEngine, window: int = 10) -> np.ndarray:
    """
    Returns R [n] reliability index per person.
    Returns ones if fewer than 2 ticks have been recorded.
    """
    if len(engine._history) < 2:
        return np.ones(engine.n, dtype=np.float32)

    recent = engine._history[-window:]
    V_series = np.stack([s["V"] for s in recent])   # [T, n]
    delta_V  = np.abs(np.diff(V_series, axis=0))     # [T-1, n]
    mean_dV  = delta_V.mean(axis=0)                  # [n]
    V_now    = V_series[-1]                           # [n]
    R = 1.0 - mean_dV / (V_now + 1e-6)
    return np.clip(R, 0.0, 1.0).astype(np.float32)


def high_entropy_people(engine: DRVEngine, window: int = 10) -> list[str]:
    """Return names of people whose reliability is below threshold."""
    R = compute_reliability(engine, window)
    return [engine.names[i] for i in np.where(R < R_THRESHOLD)[0]]


def back_calculate_eta(engine: DRVEngine) -> np.ndarray:
    """
    η'_{i,j} = C_{i,j} / (CAP_i · p_{i,j} · W_j)
    Returns [n,m]; undefined cells (denom=0) filled with 1.0.
    """
    denom = engine.CAP[:, None] * engine.P * engine.W[None, :]
    C     = engine.contribution()
    return np.where(denom > 1e-8, C / denom, 1.0).astype(np.float32)


def apply_eta_feedback(engine: DRVEngine, alpha: float = 0.1):
    """
    One-step η adaptation: pull H toward back-calculated η'.
    α controls how fast the efficiency matrix tracks actual performance.
    """
    eta_back  = back_calculate_eta(engine)
    engine.H += alpha * (eta_back - engine.H)
    engine.H  = np.clip(engine.H, 0.1, 3.0)


def value_history(engine: DRVEngine) -> np.ndarray:
    """
    Returns V_series [T, n] from history.
    Shape is (0, n) if no history yet.
    """
    if not engine._history:
        return np.zeros((0, engine.n), dtype=np.float32)
    return np.stack([s["V"] for s in engine._history])


def flow_history(engine: DRVEngine) -> np.ndarray:
    """Returns flow_series [T, m] from history."""
    if not engine._history:
        return np.zeros((0, engine.m), dtype=np.float32)
    return np.stack([s["flow"] for s in engine._history])
