"""
build_simgraph.py — Factory that constructs a SimGraph from DRVE data tensors.

Usage:
    from build_simgraph import build_from_tensors
    sim, person_ids, proj_keys = build_from_tensors(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)
"""

import numpy as np
from simgraph import SimGraph
from nodes import PersonNode, ProjectNode


def build_from_tensors(P, CAP, W_arr, X, H, G_arr, PHI, ALPHA, names, proj_keys):
    """
    P           [n,m]   allocation fractions (rows sum to 1)
    CAP         [n]     person capability scores
    W_arr       [m]     project resistance
    X           [n,m]   eligibility mask
    H           [n,m]   base eta matrix (skill match, ignored — ALPHA handles boosts)
    G_arr       [m]     project goals
    PHI         [m,m]   progress gating  — PHI[j,k] means k gates j
    ALPHA       [m,m]   efficiency boost — ALPHA[j,k] means k boosts eta on j
    names       [n]     person names
    proj_keys   [m]     project string keys
    """
    n, m = P.shape
    sim  = SimGraph()

    # ── Person nodes ──────────────────────────────────────────────────────────
    # Convert P[i] (fractions) to allocs dict with percentages
    for i, name in enumerate(names):
        allocs = {
            proj_keys[j]: float(P[i, j] * 100)
            for j in range(m)
            if X[i, j] > 0 and P[i, j] > 0.001
        }
        node = PersonNode(
            f"p_{i}",
            params={
                "CAP":    float(CAP[i]),
                "allocs": allocs,
                "name":   name,
                "frozen": False,
            },
        )
        sim.add(node)

    # ── Project nodes ─────────────────────────────────────────────────────────
    for j, key in enumerate(proj_keys):
        sim.add(ProjectNode(
            key,
            params={
                "W":     float(W_arr[j]),
                "G":     float(G_arr[j]),
                "label": key,
            },
        ))

    # ── Person → Project contribution edges ───────────────────────────────────
    for i in range(n):
        for j in range(m):
            if X[i, j] > 0 and P[i, j] > 0.001:
                sim.connect(f"p_{i}", "contribution", proj_keys[j], "contribution")

    # ── PHI: progress gating edges ─────────────────────────────────────────────
    # PHI[j,k] > 0  →  project k gates project j
    for j in range(m):
        for k in range(m):
            phi = float(PHI[j, k])
            if phi > 1e-6:
                # Edge: proj_k.readiness → proj_j.gate, with transform s^phi
                phi_captured = phi
                sim.connect(
                    proj_keys[k], "readiness",
                    proj_keys[j], "gate",
                    transform=lambda s, φ=phi_captured: max(float(s), 1e-6) ** φ,
                    label=f"PHI[{j},{k}]={phi:.2f}",
                )

    # ── ALPHA: efficiency boost edges ──────────────────────────────────────────
    # ALPHA[j,k] > 0  →  project k boosts eta on project j's contributors.
    # Emit log(1 + alpha*S) so multiple boosts sum in log space; PersonNode
    # exponentiates to get the compounded factor = product of (1+alpha_k*S_k).
    import math as _math
    for j in range(m):
        for k in range(m):
            alpha_val = float(ALPHA[j, k])
            if alpha_val < 1e-6:
                continue
            a_c = alpha_val
            for i in range(n):
                if X[i, j] > 0 and P[i, j] > 0.001:
                    sim.connect(
                        proj_keys[k], "readiness",
                        f"p_{i}", "eta_boost",
                        transform=lambda s, α=a_c: _math.log1p(α * max(float(s), 0.0)),
                        label=f"ALPHA[{j},{k}]={alpha_val:.2f}→p_{i}",
                    )

    return sim, [f"p_{i}" for i in range(n)], list(proj_keys)
