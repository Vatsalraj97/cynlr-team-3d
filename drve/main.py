"""
main.py — DRVE simulation entry point.

Experiments:
  steady    — baseline, no governor, watch steady-state values
  governor  — governor drives toward 20% above current flow
  shock     — goal shock at t=50, priority shift at t=100

Usage:
  python main.py
  python main.py --experiment governor --ticks 300 --lr 0.005
  python main.py --experiment shock --ticks 200
  python main.py --cuda          # uses GPU if available (RTX 2080 etc.)
"""
import argparse
import time
import os
import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from rich.live    import Live
from rich.console import Console

from data     import build_tensors
from engine   import DRVEngine
from governor import governor_step, eg_step, lpp_solve
from feedback import compute_reliability, high_entropy_people
from events   import EventQueue
from ui       import render, CONSOLE


def parse_args():
    p = argparse.ArgumentParser(description="CynLr DRVE Simulation Engine")
    p.add_argument("--experiment", choices=["steady", "governor", "shock"], default="steady")
    p.add_argument("--ticks",      type=int,   default=200)
    p.add_argument("--dt",         type=float, default=1.0)
    p.add_argument("--lr",         type=float, default=0.008,  help="Governor learning rate")
    p.add_argument("--refresh",    type=float, default=0.08,   help="UI refresh interval (s)")
    p.add_argument("--cuda",       action="store_true",        help="Use CUDA if available")
    p.add_argument("--no-ui",      action="store_true",        help="Headless mode, print summary only")
    return p.parse_args()


def maybe_cuda(args) -> str:
    if not args.cuda:
        return "cpu"
    if not _TORCH_AVAILABLE:
        CONSOLE.print("[yellow]PyTorch not installed — CUDA unavailable, running on CPU[/]")
        return "cpu"
    if not torch.cuda.is_available():
        CONSOLE.print("[yellow]CUDA not available on this machine — running on CPU[/]")
        return "cpu"
    name = torch.cuda.get_device_name(0)
    CONSOLE.print(f"[green]CUDA enabled: {name}[/]")
    return "cuda"


def build_engine(args) -> DRVEngine:
    P, CAP, W, X, H, G, names, proj_keys = build_tensors()

    # Bootstrap goals: run one tick at current allocation, set G = flow × 50
    bootstrap = DRVEngine(P, CAP, W, X, H, G, names, proj_keys)
    s0 = bootstrap.tick()
    G  = np.maximum(s0["flow"] * 50.0, 1.0)   # 50-tick goal per project

    return DRVEngine(P, CAP, W, X, H, G, names, proj_keys)


def wire_events(events: EventQueue, experiment: str, engine: DRVEngine):
    if experiment == "shock":
        events.schedule(50,  "GOAL_SHOCK",     {"factor": 2.0})
        events.schedule(100, "PRIORITY_SHIFT",  {"project": "Audi(Door Assembly)", "new_weight": 3.0})
        events.schedule(150, "EFFICIENCY_LEAK", {"person": "Mohit Jani", "factor": 0.5})
        events.schedule(175, "PERSON_FREEZE",   {"person": "Sanjith Sudhakar"})
        events.schedule(185, "PERSON_THAW",     {"person": "Sanjith Sudhakar"})


def print_summary(engine: DRVEngine, state: dict, reliability: np.ndarray):
    C    = state["C"]
    flow = state["flow"]
    V    = state["V"]
    t    = state["t"]

    CONSOLE.rule("[cyan]DRVE Final Summary[/]")
    CONSOLE.print(f"t={t:.0f}  V_total={flow.sum():.1f}  avg_util={(V/np.maximum(engine.CAP,1e-6)).mean()*100:.0f}%")

    CONSOLE.print("\n[bold]Top 5 Projects by Value Flow:[/]")
    top_j = np.argsort(flow)[::-1][:5]
    for j in top_j:
        CONSOLE.print(f"  {engine.proj_keys[j]:<35} {flow[j]:.1f}  goal {engine.A[j]/engine.G[j]*100:.0f}%")

    CONSOLE.print("\n[bold]Top 5 People by Value Output:[/]")
    top_i = np.argsort(V)[::-1][:5]
    for i in top_i:
        util = V[i] / max(engine.CAP[i], 1e-6)
        CONSOLE.print(f"  {engine.names[i]:<35} V={V[i]:.1f}  util={util*100:.0f}%  R={reliability[i]:.2f}")

    he = high_entropy_people(engine)
    if he:
        CONSOLE.print(f"\n[bold red]High-Entropy People (R < 0.7):[/] {', '.join(he)}")

    CONSOLE.print("\n[bold]LPP Optimum vs Current:[/]")
    P_opt    = lpp_solve(engine)
    snap     = engine.snapshot()
    engine.P = P_opt
    opt_C    = engine.contribution()
    opt_flow = opt_C.sum(axis=0)
    engine.restore(snap)
    CONSOLE.print(f"  Current V_total : {flow.sum():.1f}")
    CONSOLE.print(f"  Optimal V_total : {opt_flow.sum():.1f}")
    CONSOLE.print(f"  Gap             : {opt_flow.sum() - flow.sum():.1f}  ({(opt_flow.sum()/flow.sum()-1)*100:.1f}% upside)")


def run(args):
    engine = build_engine(args)
    events = EventQueue()
    wire_events(events, args.experiment, engine)

    use_governor = args.experiment == "governor"
    V_target     = None
    if use_governor:
        # Target = 20% above steady-state flow
        s_init   = engine.tick()
        V_target = s_init["flow"] * 1.20
        engine.reset_accumulator()

    CONSOLE.print(f"\n[bold cyan]DRVE[/] — experiment=[bold]{args.experiment}[/]  ticks={args.ticks}  lr={args.lr}")
    CONSOLE.print(f"n={engine.n} people  m={engine.m} projects  dt={args.dt}\n")

    reliability  = np.ones(engine.n, dtype=np.float32)
    last_state   = None

    if args.no_ui:
        for tick in range(args.ticks):
            fired = events.process(engine, engine.t)
            state = engine.tick(args.dt)
            if use_governor and V_target is not None:
                governor_step(engine, V_target, lr=args.lr)
            reliability = compute_reliability(engine)
        print_summary(engine, state, reliability)
        return

    with Live(console=CONSOLE, refresh_per_second=int(1 / args.refresh)) as live:
        for tick in range(args.ticks):
            fired  = events.process(engine, engine.t)
            state  = engine.tick(args.dt)

            if use_governor and V_target is not None:
                governor_step(engine, V_target, lr=args.lr)

            reliability = compute_reliability(engine)
            renderable  = render(engine, state, reliability, fired, events.log())
            live.update(renderable)

            last_state = state
            time.sleep(args.refresh)

    if last_state:
        print_summary(engine, last_state, reliability)


if __name__ == "__main__":
    args = parse_args()
    _ = maybe_cuda(args)   # inform but numpy path is always used; torch path for future
    run(args)
