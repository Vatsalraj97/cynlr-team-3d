"""
server.py — DRVE simulation WebSocket + HTTP server.

Protocol (server → client):
  { type:"tick", t, flow[M], flow_max, cij_max, S[M], V[N], C_flat[N*M], W[M], paused, governor, tps }

Protocol (client → server):
  { type:"set_weight",      j, value }
  { type:"set_tps",         value }
  { type:"set_lr",          value }
  { type:"toggle_governor" }
  { type:"toggle_pause" }
  { type:"reset" }
  { type:"fire_event",      event, payload }
"""
import asyncio
import json
import os
import sys
import numpy as np
from aiohttp import web, WSMsgType

# ── resolve paths when running as PyInstaller bundle ──────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, "static")

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from data              import build_tensors
from engine            import DRVEngine
from simgraph_adapter  import SimGraphAdapter
from governor          import governor_step
from events            import EventQueue

# Toggle: set USE_SIMGRAPH=true env var to route through the flexible engine.
USE_SIMGRAPH = os.environ.get("USE_SIMGRAPH", "true").lower() in ("1", "true", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

def _build_engine():
    P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys = build_tensors()
    # Bootstrap: run one tick to get flow estimates, then set G to 50× that
    bootstrap = DRVEngine(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)
    s0 = bootstrap.tick()
    G  = np.maximum(s0["flow"] * 50.0, 1.0)

    if USE_SIMGRAPH:
        return SimGraphAdapter(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)
    return DRVEngine(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)


def _fresh_state():
    engine = _build_engine()
    return {
        "engine":   engine,
        "events":   EventQueue(),
        "paused":   False,
        "governor": False,
        "tps":      3,
        "dt":       1.0,
        "lr":       0.008,
        "V_target": None,
    }


STATE    = _fresh_state()
CLIENTS  = set()   # active WebSocketResponse objects
CMD_Q    = None    # asyncio.Queue, created in main()


# ─────────────────────────────────────────────────────────────────────────────
# Message building
# ─────────────────────────────────────────────────────────────────────────────

def build_tick_msg(state: dict) -> dict:
    engine = state["engine"]
    C      = engine.contribution()
    flow   = C.sum(axis=0)
    V      = C.sum(axis=1)
    R      = engine.readiness()
    B      = engine.boost()
    S      = np.clip(engine.A / np.maximum(engine.G, 1e-9), 0, 2)

    flow_max = max(float(flow.max()), 1.0)
    cij_max  = max(float(C.max()),   1.0)

    return {
        "type":      "tick",
        "t":         round(float(engine.t), 2),
        "flow":      [round(float(x), 2) for x in flow],
        "flow_max":  round(flow_max, 2),
        "cij_max":   round(cij_max, 2),
        "S":         [round(float(x), 4) for x in S],
        "R":         [round(float(x), 4) for x in R],
        "B":         [round(float(x), 4) for x in B],
        "V":         [round(float(x), 2) for x in V],
        "C_flat":    [round(float(x), 2) for x in C.flatten()],
        "W":         [round(float(x), 3) for x in engine.W],
        "G":         [round(float(x), 1) for x in engine.G],
        "PHI_flat":  [round(float(x), 3) for x in engine.PHI.flatten()],
        "ALPHA_flat":[round(float(x), 3) for x in engine.ALPHA.flatten()],
        "paused":    state["paused"],
        "governor":  state["governor"],
        "tps":       state["tps"],
        "lr":        state["lr"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Command processing
# ─────────────────────────────────────────────────────────────────────────────

def process_cmd(cmd: dict, state: dict):
    engine = state["engine"]
    t      = cmd.get("type", "")

    if t == "set_weight":
        j = int(cmd["j"])
        v = float(np.clip(cmd["value"], 0.05, 5.0))
        if 0 <= j < engine.m:
            engine.W[j] = v
            # Refresh governor target with new weights if active
            if state["governor"] and state["V_target"] is not None:
                state["V_target"] = engine.contribution().sum(axis=0) * 1.2

    elif t == "set_resistance":          # rename from set_weight — W is now resistance
        j = int(cmd["j"])
        v = float(np.clip(cmd["value"], 0.05, 10.0))
        if 0 <= j < engine.m:
            engine.W[j] = v

    elif t == "set_goal":
        j = int(cmd["j"])
        v = float(np.clip(cmd["value"], 1.0, 1e7))
        if 0 <= j < engine.m:
            engine.G[j] = v

    elif t == "set_phi":
        j = int(cmd["j"]); k = int(cmd["k"])
        v = float(np.clip(cmd["value"], 0.0, 1.0))
        if 0 <= j < engine.m and 0 <= k < engine.m:
            engine.PHI[j, k] = v

    elif t == "set_alpha":
        j = int(cmd["j"]); k = int(cmd["k"])
        v = float(np.clip(cmd["value"], 0.0, 2.0))
        if 0 <= j < engine.m and 0 <= k < engine.m:
            engine.ALPHA[j, k] = v

    elif t == "set_tps":
        state["tps"] = int(np.clip(cmd["value"], 1, 60))

    elif t == "set_lr":
        state["lr"] = float(np.clip(cmd["value"], 0.001, 0.1))

    elif t == "toggle_governor":
        state["governor"] = not state["governor"]
        if state["governor"]:
            state["V_target"] = engine.contribution().sum(axis=0) * 1.2

    elif t == "toggle_pause":
        state["paused"] = not state["paused"]

    elif t == "reset":
        state.update(_fresh_state())

    elif t == "fire_event":
        ev   = cmd.get("event", "")
        pay  = cmd.get("payload", {})
        eq   = state["events"]
        eq.schedule(engine.t, ev, pay)
        eq.process(engine, engine.t)

    elif t == "set_allocation":
        i = int(cmd["i"]); j = int(cmd["j"])
        v = float(np.clip(cmd["value"], 0.0, 1.0))
        if 0 <= i < engine.n and 0 <= j < engine.m and engine.X[i, j] > 0:
            engine.P[i, j] = v
            # Re-normalise row
            row_sum = engine.P[i].sum()
            if row_sum > 1e-6:
                engine.P[i] /= row_sum
            engine.P[i] = np.maximum(engine.P[i], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation loop
# ─────────────────────────────────────────────────────────────────────────────

async def sim_loop():
    global STATE, CLIENTS, CMD_Q
    while True:
        # Drain command queue
        while not CMD_Q.empty():
            try:
                cmd = CMD_Q.get_nowait()
                process_cmd(cmd, STATE)
            except asyncio.QueueEmpty:
                break

        if not STATE["paused"]:
            # Process scheduled events
            STATE["events"].process(STATE["engine"], STATE["engine"].t)
            # Tick
            STATE["engine"].tick(STATE["dt"])
            # Governor step
            if STATE["governor"] and STATE["V_target"] is not None:
                governor_step(STATE["engine"], STATE["V_target"], lr=STATE["lr"])
            # Broadcast
            if CLIENTS:
                msg = json.dumps(build_tick_msg(STATE))
                dead = set()
                for ws in list(CLIENTS):
                    try:
                        await ws.send_str(msg)
                    except Exception:
                        dead.add(ws)
                CLIENTS -= dead

        await asyncio.sleep(1.0 / max(STATE["tps"], 1))


# ─────────────────────────────────────────────────────────────────────────────
# HTTP + WebSocket handlers
# ─────────────────────────────────────────────────────────────────────────────

async def index_handler(request):
    path = os.path.join(STATIC_DIR, "viz.html")
    return web.FileResponse(path)


async def ws_handler(request):
    global CMD_Q
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)

    CLIENTS.add(ws)
    # Send initial state immediately
    try:
        await ws.send_str(json.dumps(build_tick_msg(STATE)))
    except Exception:
        pass

    async for msg in ws:
        if msg.type == WSMsgType.TEXT:
            try:
                cmd = json.loads(msg.data)
                await CMD_Q.put(cmd)
            except Exception:
                pass
        elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
            break

    CLIENTS.discard(ws)
    return ws


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

VIZ_DIR = os.path.join(BASE_DIR, "viz")


async def meta_handler(request):
    from data import PEOPLE, PROJECTS
    projects = [
        {"j": j, "key": k, "label": k, "W": float(w)}
        for j, (k, w) in enumerate(PROJECTS)
    ]
    people = [
        {"i": i, "name": p["name"], "level": p["level"], "allocs": p["allocs"]}
        for i, p in enumerate(PEOPLE)
    ]
    return web.json_response({
        "projects": projects,
        "people":   people,
        "n": len(PEOPLE),
        "m": len(PROJECTS),
    }, headers={"Access-Control-Allow-Origin": "*"})


async def viz_handler(request):
    path = os.path.join(VIZ_DIR, "index.html")
    return web.FileResponse(path)


async def control_handler(request):
    path = os.path.join(VIZ_DIR, "control.html")
    return web.FileResponse(path)


async def create_app():
    global CMD_Q
    CMD_Q = asyncio.Queue()

    app = web.Application()
    app.router.add_get("/",        index_handler)
    app.router.add_get("/ws",      ws_handler)
    app.router.add_get("/meta",    meta_handler)
    app.router.add_get("/viz",     viz_handler)
    app.router.add_get("/control", control_handler)
    app.router.add_static("/static", STATIC_DIR)
    if os.path.isdir(VIZ_DIR):
        app.router.add_static("/viz-assets", VIZ_DIR)

    # Start sim loop as background task
    asyncio.ensure_future(sim_loop())
    return app


def run(host="127.0.0.1", port=8765):
    web.run_app(create_app(), host=host, port=port, print=lambda s: print(s))
