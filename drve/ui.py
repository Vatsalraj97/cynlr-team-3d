"""
ui.py — Rich terminal display for DRVE simulation.

Panels:
  Header bar  — t, V_total, avg_util, active events
  Projects    — top-N by flow, with goal progress bar
  People      — top-N by V, with reliability flag
  Events log  — recent fired events
"""
import numpy as np
from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.columns import Columns
from rich.text    import Text
from rich.layout  import Layout
from rich         import box

from engine   import DRVEngine
from feedback import R_THRESHOLD

CONSOLE = Console()

_BAR_WIDTH   = 12
_TOP_PROJ    = 14
_TOP_PEOPLE  = 16


def _bar(fraction: float, width: int = _BAR_WIDTH) -> str:
    filled = min(int(fraction * width), width)
    return "█" * filled + "░" * (width - filled)


def _util_color(util: float) -> str:
    if util >= 1.5:  return "bold green"
    if util >= 1.0:  return "green"
    if util >= 0.7:  return "yellow"
    return "red"


def _r_color(r: float) -> str:
    if r >= 0.85:  return "green"
    if r >= R_THRESHOLD:  return "yellow"
    return "bold red"


def build_header(engine: DRVEngine, state: dict, events_fired: list) -> Panel:
    C     = state["C"]
    flow  = state["flow"]
    V     = state["V"]
    t     = state["t"]
    V_tot = flow.sum()
    avg_util = (V / np.maximum(engine.CAP, 1e-6)).mean()
    active_proj = int((flow > 0).sum())

    event_str = ""
    if events_fired:
        event_str = "  [bold yellow]⚡ " + " | ".join(e[1] for e in events_fired) + "[/]"

    txt = (
        f"[bold cyan]DRVE[/]  t={t:.1f}  "
        f"V_total=[bold]{V_tot:.1f}[/]  "
        f"avg_util=[bold]{avg_util*100:.0f}%[/]  "
        f"active_proj={active_proj}/{engine.m}"
        f"{event_str}"
    )
    return Panel(Text.from_markup(txt), border_style="cyan", padding=(0, 1))


def build_projects_table(engine: DRVEngine, state: dict) -> Table:
    flow = state["flow"]
    S    = state["S"]

    t = Table(
        "Project", "W", "Flow", "Accum", "Goal%", "",
        box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan",
        min_width=60,
    )

    order = np.argsort(flow)[::-1][:_TOP_PROJ]
    for j in order:
        key   = engine.proj_keys[j]
        label = key[:22]
        pct   = S[j] * 100
        bar   = _bar(S[j])
        color = "green" if pct >= 100 else ("yellow" if pct >= 50 else "white")
        t.add_row(
            label,
            f"{engine.W[j]:.1f}",
            f"[bold]{flow[j]:.1f}[/]",
            f"{engine.A[j]:.0f}",
            f"[{color}]{pct:.0f}%[/]",
            f"[{color}]{bar}[/]",
        )
    return t


def build_people_table(engine: DRVEngine, state: dict, reliability: np.ndarray) -> Table:
    V   = state["V"]
    C   = state["C"]

    t = Table(
        "Person", "Lvl", "CAP", "V/tick", "Util", "R",
        box=box.SIMPLE_HEAD, show_header=True, header_style="bold magenta",
        min_width=60,
    )

    order = np.argsort(V)[::-1][:_TOP_PEOPLE]
    levels_short = {
        "Principal Engineer": "PE", "Senior Engineer": "SE", "Lead": "L",
        "Engineer 3": "E3", "Engineer 2": "E2", "Engineer 1": "E1",
        "Engineer Intern": "EI", "Intern": "In",
    }
    from data import PEOPLE
    level_map = {p["name"]: p["level"] for p in PEOPLE}

    for i in order:
        name  = engine.names[i].split()[0]
        level = levels_short.get(level_map.get(engine.names[i], ""), "?")
        util  = V[i] / max(engine.CAP[i], 1e-6)
        R     = reliability[i]
        flag  = " ⚠" if R < R_THRESHOLD else ""
        t.add_row(
            name,
            level,
            f"{engine.CAP[i]:.0f}",
            f"[{_util_color(util)}]{V[i]:.1f}[/]",
            f"[{_util_color(util)}]{util*100:.0f}%[/]",
            f"[{_r_color(R)}]{R:.2f}{flag}[/]",
        )
    return t


def build_events_panel(events_log: list, max_rows: int = 5) -> Panel:
    recent = events_log[-max_rows:]
    lines  = [f"[dim]{t:.0f}[/]  [yellow]{etype}[/]  {payload}" for t, etype, payload in recent]
    body   = "\n".join(lines) if lines else "[dim]no events yet[/]"
    return Panel(body, title="Events", border_style="yellow", padding=(0, 1))


def render(
    engine:       DRVEngine,
    state:        dict,
    reliability:  np.ndarray,
    events_fired: list,
    events_log:   list,
) -> list:
    """Return a list of Rich renderables for a Live.update() call."""
    header   = build_header(engine, state, events_fired)
    proj_tbl = build_projects_table(engine, state)
    peop_tbl = build_people_table(engine, state, reliability)
    cols     = Columns([
        Panel(proj_tbl, title="[cyan]Projects[/]",  border_style="dim cyan"),
        Panel(peop_tbl, title="[magenta]People[/]", border_style="dim magenta"),
    ], equal=False)
    ev_panel = build_events_panel(events_log)

    from rich.console import Group
    return Group(header, cols, ev_panel)
