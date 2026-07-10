"""
nodes.py — Built-in node types for DRVE.

Each class is a self-contained equation bundle.
Add a new behaviour by subclassing Node and implementing tick().

Available nodes:
    PersonNode      — emits C[j] per project based on CAP × p × eta
    ProjectNode     — accumulates FLOW/W × R, tracks S = A/G
    ToolNode        — produces an eta boost signal proportional to its own S
    MarketNode      — external demand signal that modulates project W
    EventNode       — one-shot or periodic impulse injected into the graph
"""

import math
import numpy as np
from simgraph import Node


# ─────────────────────────────────────────────────────────────────────────────
# PersonNode
# ─────────────────────────────────────────────────────────────────────────────

class PersonNode(Node):
    """
    Models one engineer / contributor.

    Params:
        CAP     float   capability score (set by seniority)
        allocs  dict    {project_id: pct}  — raw % values, engine normalises
        frozen  bool    if True, outputs zero (PERSON_FREEZE event)

    Input ports:
        eta_boost   float   global efficiency multiplier (from ToolNodes)

    Output ports:
        contribution    dict    {project_id: C_value}
        V               float   total value output this tick
        utilisation     float   V / CAP
    """

    def input_ports(self):
        return ["eta_boost"]

    def output_ports(self):
        return ["contribution", "V", "utilisation"]

    def tick(self, dt: float) -> None:
        if self.params.get("frozen", False):
            self.emit("contribution", {})
            self.emit("V", 0.0)
            self.emit("utilisation", 0.0)
            return

        cap    = self.params.get("CAP", 60.0)
        allocs = self.params.get("allocs", {})
        total  = sum(allocs.values()) or 1.0
        # eta_boost edges emit log(1+alpha*S) so multiple boosts can be summed
        # and exponentiated here — gives product of factors without a special aggregator
        log_eta = self.get("eta_boost", 0.0)
        eta     = math.exp(float(log_eta))

        contrib = {
            proj_id: cap * (pct / total) * eta
            for proj_id, pct in allocs.items()
            if pct > 0
        }

        V = sum(contrib.values())
        self.state["V"]    = V
        self.state["util"] = V / max(cap, 1e-6)

        self.emit("contribution",  contrib)
        self.emit("V",             V)
        self.emit("utilisation",   self.state["util"])


# ─────────────────────────────────────────────────────────────────────────────
# ProjectNode
# ─────────────────────────────────────────────────────────────────────────────

class ProjectNode(Node):
    """
    Accumulates progress from people, gated by upstream project readiness.

    dA/dt = (FLOW / W) × R

    Params:
        W       float   resistance — difficulty (higher = slower progress)
        G       float   goal — total energy required to complete
        label   str     display name

    Input ports:
        contribution    dict    {project_id: C_value} from each PersonNode
        gate            float   readiness signal from upstream projects
                                (multiple edges are MULTIPLIED together)

    Output ports:
        S               float   completion fraction A/G  ∈ [0,1]
        flow            float   total energy arriving this tick
        rate            float   dA/dt — actual progress rate
        readiness       float   = S, passed to downstream gates
    """

    def input_ports(self):
        return ["contribution", "gate"]

    def output_ports(self):
        return ["S", "flow", "rate", "readiness"]

    def tick(self, dt: float) -> None:
        W = max(self.params.get("W", 1.0), 1e-3)
        G = max(self.params.get("G", 1000.0), 1e-6)

        # Sum contributions addressed to this project
        raw = self.inp.get("contribution", {})
        if isinstance(raw, dict):
            flow = raw.get(self.id, 0.0)
        else:
            flow = float(raw or 0.0)

        # Readiness gate — all incoming gate signals multiplied
        # (each gate edge carries S[k]^PHI[j,k] via its transform lambda)
        gates = self.inp.get("gate")
        if gates is None:
            R = 1.0
        elif isinstance(gates, list):
            R = math.prod(max(g, 0.0) for g in gates)
        else:
            R = max(float(gates), 0.0)

        dA = (flow / W) * R * dt
        self.state["A"]  = self.state.get("A", 0.0) + dA
        S = min(self.state["A"] / G, 1.0)
        self.state["S"]  = S
        self.state["R"]  = R

        self.emit("S",          S)
        self.emit("flow",       flow)
        self.emit("rate",       dA / max(dt, 1e-9))
        self.emit("readiness",  S)


# ─────────────────────────────────────────────────────────────────────────────
# ToolNode
# ─────────────────────────────────────────────────────────────────────────────

class ToolNode(Node):
    """
    A project that boosts efficiency of downstream nodes as it matures.

    This is a ProjectNode that also emits an eta_boost signal.
    eta_boost = 1 + alpha × S

    Params:
        W, G    same as ProjectNode
        alpha   float   boost coefficient (how much a full S helps)

    Output ports:
        (all ProjectNode ports) +
        eta_boost   float   efficiency multiplier for connected people/projects
    """

    def input_ports(self):
        return ["contribution", "gate"]

    def output_ports(self):
        return ["S", "flow", "rate", "readiness", "eta_boost"]

    def tick(self, dt: float) -> None:
        W = max(self.params.get("W", 1.0), 1e-3)
        G = max(self.params.get("G", 1000.0), 1e-6)

        raw  = self.inp.get("contribution", {})
        flow = raw.get(self.id, 0.0) if isinstance(raw, dict) else float(raw or 0.0)

        gates = self.inp.get("gate")
        R = 1.0
        if gates is not None:
            R = math.prod(max(g, 0.0) for g in (gates if isinstance(gates, list) else [gates]))

        dA = (flow / W) * R * dt
        self.state["A"] = self.state.get("A", 0.0) + dA
        S = min(self.state["A"] / G, 1.0)
        self.state["S"] = S

        alpha     = self.params.get("alpha", 0.3)
        eta_boost = 1.0 + alpha * S

        self.emit("S",          S)
        self.emit("flow",       flow)
        self.emit("rate",       dA / max(dt, 1e-9))
        self.emit("readiness",  S)
        self.emit("eta_boost",  eta_boost)


# ─────────────────────────────────────────────────────────────────────────────
# MarketNode
# ─────────────────────────────────────────────────────────────────────────────

class MarketNode(Node):
    """
    External demand / pressure signal. No inputs — driven by params or events.

    Emits a 'demand' signal [0,1] that other nodes can read.
    Example use: connect to ProjectNode gate to model customer urgency
    driving up effective priority.

    Params:
        demand          float   current demand level [0,1]
        growth_rate     float   demand grows by this much per tick (default 0)
        decay_rate      float   demand decays by this much per tick (default 0)
        cap             float   maximum demand (default 1.0)
    """

    def input_ports(self):
        return []

    def output_ports(self):
        return ["demand", "urgency"]

    def tick(self, dt: float) -> None:
        d     = self.state.get("demand", self.params.get("demand", 0.5))
        grow  = self.params.get("growth_rate", 0.0)
        decay = self.params.get("decay_rate",  0.0)
        cap   = self.params.get("cap", 1.0)

        d = min(d + grow * dt - decay * dt, cap)
        d = max(d, 0.0)
        self.state["demand"] = d

        self.emit("demand",   d)
        self.emit("urgency",  d)   # alias — lets you connect to different ports


# ─────────────────────────────────────────────────────────────────────────────
# BurnoutNode  (wraps a person, adds a degradation equation)
# ─────────────────────────────────────────────────────────────────────────────

class BurnoutPersonNode(PersonNode):
    """
    PersonNode that degrades CAP when utilisation stays above threshold.

    Extra params:
        burnout_threshold   float   util level that starts burnout (default 1.1)
        burnout_rate        float   CAP lost per tick when overloaded (default 0.5)
        recovery_rate       float   CAP recovered per tick when underloaded (default 0.2)
        cap_floor           float   minimum CAP (default 20)
        cap_ceiling         float   maximum CAP, used for recovery (default original CAP)
    """

    def tick(self, dt: float) -> None:
        super().tick(dt)

        cap       = self.params.get("CAP", 60.0)
        threshold = self.params.get("burnout_threshold", 1.1)
        b_rate    = self.params.get("burnout_rate",  0.5)
        r_rate    = self.params.get("recovery_rate", 0.2)
        floor     = self.params.get("cap_floor",     20.0)
        ceiling   = self.params.get("cap_ceiling",   cap)

        util = self.state.get("util", 0.0)
        current_cap = self.params.get("CAP", cap)

        if util > threshold:
            new_cap = max(current_cap - b_rate * dt, floor)
            self.state["burnout_ticks"] = self.state.get("burnout_ticks", 0) + 1
        else:
            new_cap = min(current_cap + r_rate * dt, ceiling)
            self.state["burnout_ticks"] = 0

        self.params["CAP"] = new_cap
        self.state["cap_current"] = new_cap


# ─────────────────────────────────────────────────────────────────────────────
# EventNode  (one-shot or periodic impulse)
# ─────────────────────────────────────────────────────────────────────────────

class EventNode(Node):
    """
    Emits an impulse at specified tick(s). Connect to any input port.

    Params:
        fire_at     list[float]     simulation times to fire
        value       any             value to emit when firing
        port        str             output port name (default 'impulse')
        repeat      float           if set, fires every N ticks (overrides fire_at)
    """

    def input_ports(self):
        return []

    def output_ports(self):
        return [self.params.get("port", "impulse")]

    def tick(self, dt: float) -> None:
        port    = self.params.get("port", "impulse")
        value   = self.params.get("value", 1.0)
        repeat  = self.params.get("repeat")
        fire_at = self.params.get("fire_at", [])
        t       = self.state.get("t", 0.0) + dt
        self.state["t"] = t

        firing = False
        if repeat and t > 0:
            prev = t - dt
            firing = int(t / repeat) > int(prev / repeat)
        elif fire_at:
            firing = any(t - dt < ft <= t for ft in fire_at)

        self.emit(port, value if firing else None)
