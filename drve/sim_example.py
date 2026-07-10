"""
sim_example.py — demonstrates the SimGraph engine.

Run:  python sim_example.py
"""

from simgraph import SimGraph
from nodes import PersonNode, ProjectNode, ToolNode, BurnoutPersonNode, MarketNode

# ─── Build the system ─────────────────────────────────────────────────────────

sim = SimGraph()

# People
sim.add(PersonNode("michael", params={"CAP": 150, "allocs": {"Robotics": 75, "AMAT1": 15, "CyRo": 10}}))
sim.add(PersonNode("shushuai", params={"CAP": 120, "allocs": {"CLX": 30, "AMAT1": 20, "SW_Tools": 20, "Robotics": 30}}))
sim.add(BurnoutPersonNode("prakash", params={"CAP": 85, "allocs": {"Lam": 80, "Schneider": 5, "RE_HW": 15},
                                              "burnout_threshold": 1.05, "burnout_rate": 0.3}))

# Tool project (boosts SW projects as it matures)
sim.add(ToolNode("SW_Tools", params={"W": 1.0, "G": 3000, "alpha": 0.30, "label": "SW Tools"}))

# Platform
sim.add(ProjectNode("CLX",      params={"W": 1.6, "G": 8000,  "label": "CLX"}))

# Research
sim.add(ProjectNode("Robotics", params={"W": 1.2, "G": 5000,  "label": "Robotics Research"}))

# Products
sim.add(ProjectNode("CyRo",     params={"W": 1.8, "G": 12000, "label": "CyRo"}))

# Solutions
sim.add(ProjectNode("Lam",      params={"W": 2.0, "G": 18000, "label": "Lam Research"}))
sim.add(ProjectNode("AMAT1",    params={"W": 2.0, "G": 15000, "label": "AMAT-1"}))
sim.add(ProjectNode("Schneider",params={"W": 2.0, "G": 14000, "label": "Schneider"}))

# Market demand node — Lam's urgency is rising
sim.add(MarketNode("lam_market", params={"demand": 0.3, "growth_rate": 0.002, "cap": 1.0}))


# ─── Wire people to projects ──────────────────────────────────────────────────

sim.connect("michael",  "contribution", "Robotics",  "contribution")
sim.connect("michael",  "contribution", "AMAT1",     "contribution")
sim.connect("michael",  "contribution", "CyRo",      "contribution")

sim.connect("shushuai", "contribution", "CLX",       "contribution")
sim.connect("shushuai", "contribution", "AMAT1",     "contribution")
sim.connect("shushuai", "contribution", "SW_Tools",  "contribution")
sim.connect("shushuai", "contribution", "Robotics",  "contribution")

sim.connect("prakash",  "contribution", "Lam",       "contribution")
sim.connect("prakash",  "contribution", "Schneider", "contribution")


# ─── Wire project dependencies ───────────────────────────────────────────────
# PHI coupling lives on the edge as a transform lambda: s → s^PHI

# CLX gates CyRo (PHI=0.6)
sim.connect("CLX", "readiness", "CyRo", "gate", transform=lambda s: s**0.6)

# Robotics gates CyRo (PHI=0.3)
sim.connect("Robotics", "readiness", "CyRo", "gate", transform=lambda s: s**0.3)

# SW Tools boosts CLX engineers' efficiency (alpha=0.3 already in ToolNode)
sim.connect("SW_Tools", "eta_boost", "shushuai", "eta_boost")

# Market urgency boosts Lam's effective readiness (external pressure)
sim.connect("lam_market", "urgency", "Lam", "gate", transform=lambda d: 0.5 + 0.5*d)


# ─── Run the simulation ───────────────────────────────────────────────────────

print(f"\n{'Tick':>5}  {'CLX S':>7}  {'CyRo S':>8}  {'Lam S':>7}  {'Prakash CAP':>12}  {'SW_Tools S':>10}")
print("-" * 65)

for tick in range(200):
    sim.tick(dt=1.0)

    if tick % 20 == 0:
        clx   = sim.nodes["CLX"].out.get("S", 0)
        cyro  = sim.nodes["CyRo"].out.get("S", 0)
        lam   = sim.nodes["Lam"].out.get("S", 0)
        pcap  = sim.nodes["prakash"].params["CAP"]
        swt   = sim.nodes["SW_Tools"].out.get("S", 0)

        print(f"{tick:>5}  {clx:>7.1%}  {cyro:>8.1%}  {lam:>7.1%}  {pcap:>12.1f}  {swt:>10.1%}")


# ─── Show how easy it is to change things ─────────────────────────────────────

print("\n── At tick 200: hire a new engineer focused on CLX ──")
sim.add(PersonNode("new_hire", params={"CAP": 70, "allocs": {"CLX": 80, "SW_Tools": 20}}))
sim.connect("new_hire", "contribution", "CLX",     "contribution")
sim.connect("new_hire", "contribution", "SW_Tools", "contribution")

print("── Freeze prakash for 10 ticks ──")
sim.nodes["prakash"].params["frozen"] = True

for tick in range(200, 250):
    sim.tick(dt=1.0)
    if tick % 10 == 0:
        clx  = sim.nodes["CLX"].out.get("S", 0)
        cyro = sim.nodes["CyRo"].out.get("S", 0)
        lam  = sim.nodes["Lam"].out.get("S", 0)
        pcap = sim.nodes["prakash"].params["CAP"]
        print(f"{tick:>5}  {clx:>7.1%}  {cyro:>8.1%}  {lam:>7.1%}  {pcap:>12.1f}")

sim.nodes["prakash"].params["frozen"] = False
print("── Prakash unfrozen ──")

print(f"\nFinal graph: {sim.stats()}")
print("\nEdges:")
for e in sim.edges:
    label = f"  [{e.label}]" if e.label else ""
    print(f"  {e}")
