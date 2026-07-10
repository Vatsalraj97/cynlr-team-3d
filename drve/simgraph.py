"""
simgraph.py — Flexible simulation graph engine.

Architecture:
  - Nodes have state, params, input ports, output ports, and a tick() equation
  - Edges connect output ports to input ports, with optional transform functions
  - Engine evaluates nodes in topological order each tick
  - Add/remove nodes and edges at runtime — order is recomputed automatically

Usage:
    sim = SimGraph()
    sim.add(PersonNode("alice", params={"CAP": 120, "allocs": {"CLX": 60}}))
    sim.add(ProjectNode("CLX",  params={"W": 1.6, "G": 5000}))
    sim.connect("alice", "contribution", "CLX", "contribution")
    sim.tick(dt=1.0)
"""

from collections import defaultdict, deque
from typing import Any, Callable, Optional
import math


# ─────────────────────────────────────────────────────────────────────────────
# Node — base class
# ─────────────────────────────────────────────────────────────────────────────

class Node:
    """
    Base class for every entity in the simulation.

    Subclass and override:
        inputs()   -> list[str]   port names this node reads each tick
        outputs()  -> list[str]   port names this node writes each tick
        tick(dt)                  read self.inp, update self.state, write self.out
    """

    def __init__(self, node_id: str, params: dict = None):
        self.id     = node_id
        self.params = dict(params or {})
        self.state: dict[str, Any]  = {}   # persists across ticks
        self.inp:   dict[str, Any]  = {}   # populated by engine BEFORE tick
        self.out:   dict[str, Any]  = {}   # populated by node DURING tick

    # override these
    def input_ports(self)  -> list[str]: return []
    def output_ports(self) -> list[str]: return []
    def tick(self, dt: float) -> None:   raise NotImplementedError

    # helpers
    def set_param(self, key: str, value: Any) -> None:
        self.params[key] = value

    def get(self, port: str, default=None):
        """Read an input port with a fallback."""
        return self.inp.get(port, default)

    def emit(self, port: str, value: Any) -> None:
        """Write to an output port."""
        self.out[port] = value

    def snapshot(self) -> dict:
        return {
            "id":     self.id,
            "params": dict(self.params),
            "state":  dict(self.state),
            "out":    dict(self.out),
        }

    def __repr__(self):
        return f"{self.__class__.__name__}({self.id!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Edge — connection between nodes
# ─────────────────────────────────────────────────────────────────────────────

class Edge:
    """
    Directed connection: source.out_port → target.in_port

    Optional transform: a function applied to the value in transit.
    Example: lambda s: s**0.6  encodes PHI=0.6 readiness gating.
    """

    def __init__(
        self,
        from_id:   str,
        out_port:  str,
        to_id:     str,
        in_port:   str,
        transform: Optional[Callable] = None,
        label:     str = "",
    ):
        self.from_id   = from_id
        self.out_port  = out_port
        self.to_id     = to_id
        self.in_port   = in_port
        self.transform = transform or (lambda x: x)
        self.label     = label

    def __repr__(self):
        return f"Edge({self.from_id}.{self.out_port} → {self.to_id}.{self.in_port})"


# ─────────────────────────────────────────────────────────────────────────────
# SimGraph — the engine
# ─────────────────────────────────────────────────────────────────────────────

class SimGraph:
    """
    Owns all nodes and edges. Evaluates the system each tick.

    Key operations:
        sim.add(node)                          register a node
        sim.remove(node_id)                    deregister node + all its edges
        sim.connect(from, out, to, in, fn)     add an edge
        sim.disconnect(from, to)               remove edges
        sim.tick(dt)                           one simulation step
        sim.snapshot()                         full state dict
    """

    def __init__(self):
        self.nodes:  dict[str, Node] = {}
        self.edges:  list[Edge]      = []
        self.t:      float           = 0.0
        self._order: list[str]       = []
        self._dirty: bool            = True   # recompute topo order on next tick

    # ── graph construction ────────────────────────────────────────────────────

    def add(self, node: Node) -> "SimGraph":
        self.nodes[node.id] = node
        self._dirty = True
        return self

    def remove(self, node_id: str) -> "SimGraph":
        self.nodes.pop(node_id, None)
        self.edges = [e for e in self.edges
                      if e.from_id != node_id and e.to_id != node_id]
        self._dirty = True
        return self

    def connect(
        self,
        from_id:   str,
        out_port:  str,
        to_id:     str,
        in_port:   str,
        transform: Optional[Callable] = None,
        label:     str = "",
    ) -> "SimGraph":
        self.edges.append(Edge(from_id, out_port, to_id, in_port, transform, label))
        self._dirty = True
        return self

    def disconnect(self, from_id: str, to_id: str, out_port: str = None) -> "SimGraph":
        self.edges = [
            e for e in self.edges
            if not (
                e.from_id == from_id and e.to_id == to_id
                and (out_port is None or e.out_port == out_port)
            )
        ]
        self._dirty = True
        return self

    # ── evaluation ────────────────────────────────────────────────────────────

    def _topo_sort(self) -> list[str]:
        """Kahn's algorithm. Handles cycles by appending remaining nodes."""
        ids    = set(self.nodes)
        in_deg = defaultdict(int)
        adj    = defaultdict(list)

        for e in self.edges:
            if e.from_id in ids and e.to_id in ids:
                adj[e.from_id].append(e.to_id)
                in_deg[e.to_id]    += 1
                in_deg.setdefault(e.from_id, 0)
        for nid in ids:
            in_deg.setdefault(nid, 0)

        queue  = deque(n for n in ids if in_deg[n] == 0)
        order  = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in adj[n]:
                in_deg[m] -= 1
                if in_deg[m] == 0:
                    queue.append(m)

        # fallback for cycles — append remaining
        seen = set(order)
        order += [n for n in ids if n not in seen]
        return order

    def _gather(self, node_id: str) -> dict:
        """
        Collect all incoming edge values for a node.
        Multiple edges into the same port are combined:
          - numbers / arrays  → summed
          - dicts             → merged (values summed for duplicate keys)
          - lists             → concatenated
        """
        buckets: dict[str, list] = defaultdict(list)
        for e in self.edges:
            if e.to_id != node_id or e.from_id not in self.nodes:
                continue
            val = self.nodes[e.from_id].out.get(e.out_port)
            if val is not None:
                buckets[e.in_port].append(e.transform(val))

        result = {}
        for port, vals in buckets.items():
            if len(vals) == 1:
                result[port] = vals[0]
            elif isinstance(vals[0], dict):
                merged: dict = {}
                for d in vals:
                    for k, v in d.items():
                        merged[k] = merged.get(k, 0) + v
                result[port] = merged
            else:
                try:
                    result[port] = sum(vals)
                except TypeError:
                    result[port] = vals
        return result

    def tick(self, dt: float = 1.0) -> dict:
        if self._dirty:
            self._order = self._topo_sort()
            self._dirty = False

        for node_id in self._order:
            node = self.nodes[node_id]
            node.inp = self._gather(node_id)
            node.tick(dt)

        self.t += dt
        return self.snapshot()

    # ── introspection ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "t":     self.t,
            "nodes": {nid: n.snapshot() for nid, n in self.nodes.items()},
        }

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def edges_of(self, node_id: str) -> dict:
        return {
            "in":  [e for e in self.edges if e.to_id   == node_id],
            "out": [e for e in self.edges if e.from_id == node_id],
        }

    def stats(self) -> dict:
        return {
            "t":          self.t,
            "nodes":      len(self.nodes),
            "edges":      len(self.edges),
            "eval_order": list(self._order),
        }
