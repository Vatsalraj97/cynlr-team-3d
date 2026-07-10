# -*- coding: utf-8 -*-
"""
app.py - DRVE  Dynamic Resource Value Engine
Standalone native desktop app.  No browser required.
Run:  python app.py
"""
import threading, time, sys, os
import numpy as np
import dearpygui.dearpygui as dpg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import build_tensors, PROJECTS, PEOPLE
from engine import DRVEngine
from simgraph_adapter import SimGraphAdapter

# ═══════════════════════════════════════════════════════════
# PROJECT + PEOPLE METADATA
# ═══════════════════════════════════════════════════════════
M, N = 22, 53

SHORT = [
    'Audi','Lam Res','AMAT-1','AMAT-2','Schneider','Hyundai',
    'CyRo','CyNoid','CLX','Robotics',
    'SW Tools','HW Tools','Gripper','DevOps','Refactor',
    'Demo Dev','R&E HW','R&E SW','Neuro',
    'Packaging','Docs','Hiring',
]
CAT = [
    'solution','solution','solution','solution','solution','solution',
    'product','product','platform','research',
    'tool','tool','tool','tool','tool','internal',
    'research','research','research','internal','internal','internal',
]
CAT_C = {
    'solution': (245,158, 11,255),
    'product':  (  6,182,212,255),
    'platform': ( 59,130,246,255),
    'research': (167,139,250,255),
    'tool':     ( 34,197, 94,255),
    'internal': (148,163,184,200),
}

# ═══════════════════════════════════════════════════════════
# VISUAL CONSTANTS
# ═══════════════════════════════════════════════════════════

# PHI / ALPHA matrix
PM_CELL  = 34      # cell size px — wide enough for "0.00" in Consolas
PM_LHDR  = 110     # left header width (row labels)
PM_THDR  = 96      # top header height  (10 chars × 9px line = 90px + 6px margin)
CHAR_H   = 9       # vertical text line height — tight stacking
CHAR_SZ  = 9       # draw_text size for column label characters
ROW_SZ   = 12      # row label font size
VAL_SZ   = 11      # cell value font size

# People x Projects matrix
PP_CELL  = 28      # cell px
PP_LHDR  = 160     # person name column
PP_SZ    = 13      # person name font size
PP_VAL   = 11      # allocation % font size

# Colour palette
BG       = ( 14, 15, 22, 255)   # window background
PANEL    = ( 20, 22, 33, 255)   # panel / child window background
MAT_BG   = ( 18, 20, 30, 255)   # matrix area background
ZERO_C   = ( 26, 28, 42, 255)   # empty cell
DIAG_C   = ( 38, 42, 60, 255)   # diagonal cell
GRID     = ( 55, 60, 88, 255)   # cell border (visible on dark bg)
SEP      = ( 40, 44, 64, 255)   # separator lines
LCOL     = (180,188,210, 255)   # body text
DIM      = ( 90,100,128, 255)   # secondary text
HDR      = (210,218,240, 255)   # header text
PHI_HI   = ( 99,102,241, 255)   # max PHI cell colour (indigo)
ALP_HI   = ( 34,197, 94, 255)   # max ALPHA cell colour (green)
RUN_C    = ( 34,197, 94, 255)   # running indicator
PAU_C    = (239, 68, 68, 255)   # paused indicator
ACCENT   = ( 99,102,241, 255)   # buttons / sliders

# ═══════════════════════════════════════════════════════════
# ENGINE STATE
# ═══════════════════════════════════════════════════════════
eng  = None
lock = threading.Lock()
ctrl = {'tps': 5, 'paused': True, 'dt': 1.0}  # starts PAUSED
sim_running = False

def init_engine_with_config(W_init, g_mult, tps, dt):
    global eng
    P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys = build_tensors()
    for j in range(M):
        W[j] = float(W_init[j])
    boot = DRVEngine(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)
    s0   = boot.tick()
    G    = np.maximum(s0['flow'] * float(g_mult), 1.0)
    eng  = SimGraphAdapter(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)
    ctrl['tps'] = int(tps)
    ctrl['dt']  = float(dt)

snap = {
    't': 0.0,
    'W':    np.ones(M)*2.0,
    'PHI':  np.zeros((M,M)),
    'ALPHA':np.zeros((M,M)),
    'V':    np.zeros(N),
}

def sim_loop():
    global snap
    while True:
        if not ctrl['paused'] and eng is not None:
            with lock:
                eng.tick(ctrl['dt'])
                C = eng.contribution()
                snap = {
                    't':    eng.t,
                    'W':    eng.W.copy(),
                    'PHI':  eng.PHI.copy(),
                    'ALPHA':eng.ALPHA.copy(),
                    'V':    C.sum(axis=1).copy(),
                }
        time.sleep(1.0 / max(ctrl['tps'], 1))

# ═══════════════════════════════════════════════════════════
# COLOUR HELPERS
# ═══════════════════════════════════════════════════════════
def lerp4(v, lo, hi):
    v = max(0.0, min(1.0, float(v)))
    return tuple(int(lo[i] + (hi[i]-lo[i])*v) for i in range(4))

def phi_col(v):      return lerp4(v, ZERO_C, PHI_HI)
def alp_col(v):      return lerp4(v, ZERO_C, ALP_HI)
def people_col(pct, cat):
    t = min(pct / 100.0, 1.0)
    # Heat-map: dark → blue → teal → amber → red as investment increases
    stops = [
        (0.00, (26,  28,  42, 255)),   # empty / dark
        (0.20, (59, 130, 246, 255)),   # blue — light touch
        (0.45, ( 6, 182, 212, 255)),   # teal — moderate
        (0.70, (251,146,  60, 255)),   # amber — heavy
        (1.00, (239,  68,  68, 255)),  # red   — full commitment
    ]
    for i in range(len(stops)-1):
        t0, c0 = stops[i]
        t1, c1 = stops[i+1]
        if t <= t1:
            seg = (t - t0) / (t1 - t0)
            return lerp4(seg, c0, c1)
    return stops[-1][1]

# ═══════════════════════════════════════════════════════════
# CONFIG DIALOG
# ═══════════════════════════════════════════════════════════
def build_config_window(W_defaults):
    CW, CH = 820, 680

    with dpg.window(tag='cfg_win', no_title_bar=True, no_resize=True,
                    no_move=True, no_scrollbar=True,
                    width=CW, height=CH, pos=(0,0)):

        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=16)
            dpg.add_text("DRVE", color=ACCENT)
            dpg.add_text("   Initial Configuration", color=LCOL)
        dpg.add_spacer(height=4)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        with dpg.group(horizontal=True):
            dpg.add_spacer(width=16)

            # Left: clock + resistance sliders
            with dpg.child_window(width=380, height=CH-100, border=False):
                dpg.add_text("SIMULATION CLOCK", color=DIM)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                dpg.add_slider_int(tag='cfg_tps', label="Ticks / sec",
                                   default_value=5, min_value=1, max_value=30,
                                   width=200, format="tps = %d")
                dpg.add_spacer(height=4)
                dpg.add_slider_float(tag='cfg_dt', label="Time step",
                                     default_value=1.0, min_value=0.1, max_value=5.0,
                                     width=200, format="dt = %.1f")
                dpg.add_spacer(height=12)

                dpg.add_text("GOAL HORIZON", color=DIM)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                dpg.add_slider_float(tag='cfg_gmult', label="G multiplier",
                                     default_value=50.0, min_value=10.0, max_value=200.0,
                                     width=200, format="%.0f ticks")
                dpg.add_spacer(height=12)

                dpg.add_text("RESISTANCE   W   per project", color=DIM)
                dpg.add_text("Higher W = slower progress", color=DIM)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                for j in range(M):
                    col = CAT_C.get(CAT[j], (148,163,184,200))
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{SHORT[j]:<12}", color=col)
                        dpg.add_slider_float(tag=f'cfg_w_{j}',
                                             default_value=float(W_defaults[j]),
                                             min_value=0.05, max_value=5.0,
                                             width=180, no_input=True,
                                             format="%.2f")
                    dpg.add_spacer(height=2)

            dpg.add_spacer(width=16)

            # Right: legend
            with dpg.child_window(width=CW-440, height=CH-100, border=False):
                dpg.add_text("MATRIX GUIDE", color=DIM)
                dpg.add_separator()
                dpg.add_spacer(height=10)

                dpg.add_text("PHI  upper triangle", color=PHI_HI)
                dpg.add_spacer(height=4)
                dpg.add_text("  How much project K blocks\n  progress on project J", color=LCOL)
                dpg.add_spacer(height=4)
                dpg.add_text("  0.0  no gate", color=DIM)
                dpg.add_text("  0.5  soft slowdown", color=DIM)
                dpg.add_text("  1.0  hard block", color=DIM)
                dpg.add_spacer(height=16)

                dpg.add_text("ALPHA  lower triangle", color=ALP_HI)
                dpg.add_spacer(height=4)
                dpg.add_text("  How much project K boosts\n  efficiency on project J", color=LCOL)
                dpg.add_spacer(height=4)
                dpg.add_text("  0.0  no boost", color=DIM)
                dpg.add_text("  1.0  up to 2x faster", color=DIM)
                dpg.add_text("  2.0  up to 3x faster", color=DIM)
                dpg.add_spacer(height=16)

                dpg.add_text("PEOPLE MATRIX", color=DIM)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                dpg.add_text("  Colour = category", color=LCOL)
                dpg.add_text("  Brightness = allocation %", color=LCOL)
                dpg.add_spacer(height=8)
                for cat, col in CAT_C.items():
                    with dpg.group(horizontal=True):
                        dpg.add_text("  ■", color=col)
                        dpg.add_text(cat, color=DIM)

        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=16)
            dpg.add_button(label="     Open Simulation     ",
                           tag='btn_launch', width=CW-32, height=40,
                           callback=do_launch)


def do_launch():
    global sim_running
    tps    = dpg.get_value('cfg_tps')
    dt     = dpg.get_value('cfg_dt')
    g_mult = dpg.get_value('cfg_gmult')
    W_init = [dpg.get_value(f'cfg_w_{j}') for j in range(M)]

    for j in range(M):
        if dpg.does_item_exist(f'ws_{j}'):
            dpg.set_value(f'ws_{j}', W_init[j])
    if dpg.does_item_exist('tps_sl'):
        dpg.set_value('tps_sl', tps)

    init_engine_with_config(W_init, g_mult, tps, dt)
    ctrl['paused'] = True   # always start paused

    if not sim_running:
        threading.Thread(target=sim_loop, daemon=True).start()
        sim_running = True

    dpg.configure_item('cfg_win',  show=False)
    dpg.configure_item('main_win', show=True)
    dpg.set_primary_window('main_win', True)

    _draw_phi_alpha()
    _draw_people_header()
    _draw_people_body()


# ═══════════════════════════════════════════════════════════
# DRAW — PHI / ALPHA MATRIX
# ═══════════════════════════════════════════════════════════
def _draw_phi_alpha():
    dl  = 'dl_phi'
    if not dpg.does_item_exist(dl): return
    dpg.delete_item(dl, children_only=True)
    PHI = snap['PHI']
    ALP = snap['ALPHA']

    TW   = PM_LHDR + M*PM_CELL
    TH   = PM_THDR + M*PM_CELL

    # Dark background for the whole matrix
    dpg.draw_rectangle([0,0],[TW+10,TH+40],
                       fill=MAT_BG, color=(0,0,0,0), parent=dl)

    # ── Column headers (project names, vertical) ──────────────
    for k in range(M):
        x   = PM_LHDR + k*PM_CELL
        col = CAT_C.get(CAT[k], (148,163,184,200))
        cx  = x + PM_CELL//2 - 4
        for ci, ch in enumerate(SHORT[k][:10]):
            dpg.draw_text([cx, 4 + ci*CHAR_H], ch,
                          color=col, size=CHAR_SZ, parent=dl)

    # Divider line below header
    dpg.draw_line([PM_LHDR, PM_THDR-2], [TW, PM_THDR-2],
                  color=SEP, thickness=1, parent=dl)

    # ── Row headers (project names, horizontal) ───────────────
    for j in range(M):
        y   = PM_THDR + j*PM_CELL
        col = CAT_C.get(CAT[j], (148,163,184,200))
        dpg.draw_text([6, y + PM_CELL//2 - 6], SHORT[j][:12],
                      color=col, size=ROW_SZ, parent=dl)

    # Divider line right of row headers
    dpg.draw_line([PM_LHDR-2, PM_THDR], [PM_LHDR-2, TH],
                  color=SEP, thickness=1, parent=dl)

    # ── Cells ─────────────────────────────────────────────────
    for j in range(M):
        for k in range(M):
            x1 = PM_LHDR + k*PM_CELL
            y1 = PM_THDR + j*PM_CELL
            x2, y2 = x1+PM_CELL, y1+PM_CELL

            vy = y1 + PM_CELL//2 - VAL_SZ//2
            if j == k:
                dpg.draw_rectangle([x1,y1],[x2,y2],
                                   fill=DIAG_C, color=GRID, thickness=1, parent=dl)
                dpg.draw_text([x1 + PM_CELL//2 - 4, vy],
                              "1", color=(90,100,130,220), size=VAL_SZ,
                              parent=dl)

            elif j < k:
                v    = float(PHI[j,k])
                fill = phi_col(v)
                dpg.draw_rectangle([x1,y1],[x2,y2],
                                   fill=fill, color=GRID, thickness=1, parent=dl)
                if v > 0.01:
                    dpg.draw_text([x1+3, vy], f"{v:.2f}",
                                  color=(225,228,255,255), size=VAL_SZ, parent=dl)

            else:
                v    = float(ALP[j,k])
                fill = alp_col(v)
                dpg.draw_rectangle([x1,y1],[x2,y2],
                                   fill=fill, color=GRID, thickness=1, parent=dl)
                if v > 0.01:
                    dpg.draw_text([x1+3, vy], f"{v:.2f}",
                                  color=(190,255,210,255), size=VAL_SZ, parent=dl)

    # ── Legend below matrix ───────────────────────────────────
    ly = TH + 10
    entries = [
        (PHI_HI, "Upper triangle  PHI  --  gate: K blocks J    range 0 - 1"),
        (ALP_HI, "Lower triangle  ALPHA  --  boost: K speeds J   range 0 - 2"),
        (DIAG_C, "Diagonal  =  1   self-dependency"),
    ]
    for i,(c,lbl) in enumerate(entries):
        dpg.draw_rectangle([PM_LHDR+i*340, ly],
                           [PM_LHDR+i*340+12, ly+11],
                           fill=c, color=(0,0,0,0), parent=dl)
        dpg.draw_text([PM_LHDR+i*340+18, ly],
                      lbl, color=DIM, size=11, parent=dl)


# ═══════════════════════════════════════════════════════════
# DRAW — PEOPLE x PROJECTS MATRIX  (header + body separate)
# ═══════════════════════════════════════════════════════════

def _draw_people_header():
    """Frozen header strip — project names stacked vertically."""
    dl = 'dl_ppl_hdr'
    if not dpg.does_item_exist(dl): return
    dpg.delete_item(dl, children_only=True)

    TW = PP_LHDR + M*PP_CELL

    # Background
    dpg.draw_rectangle([0,0],[TW+20, PM_THDR+4],
                       fill=MAT_BG, color=(0,0,0,0), parent=dl)

    # Person-name column label
    dpg.draw_text([4, PM_THDR//2 - 6], "ENGINEER",
                  color=DIM, size=9, parent=dl)

    # Project column headers — vertical stacked characters
    for k in range(M):
        x   = PP_LHDR + k*PP_CELL
        col = CAT_C.get(CAT[k], (148,163,184,200))
        cx  = x + PP_CELL//2 - 4
        for ci, ch in enumerate(SHORT[k][:10]):
            dpg.draw_text([cx, 3 + ci*CHAR_H], ch,
                          color=col, size=CHAR_SZ, parent=dl)

    # Bottom divider
    dpg.draw_line([0, PM_THDR], [TW+20, PM_THDR],
                  color=SEP, thickness=2, parent=dl)
    # Right divider for name column
    dpg.draw_line([PP_LHDR-2, 0], [PP_LHDR-2, PM_THDR],
                  color=SEP, thickness=1, parent=dl)


def _draw_people_body():
    """Scrollable person rows — no headers."""
    dl = 'dl_ppl'
    if not dpg.does_item_exist(dl): return
    dpg.delete_item(dl, children_only=True)

    TW = PP_LHDR + M*PP_CELL

    dpg.draw_rectangle([0,0],[TW+20, N*PP_CELL+10],
                       fill=MAT_BG, color=(0,0,0,0), parent=dl)

    # Vertical divider for name column
    dpg.draw_line([PP_LHDR-2, 0], [PP_LHDR-2, N*PP_CELL],
                  color=SEP, thickness=1, parent=dl)

    for i, person in enumerate(PEOPLE):
        y1   = i*PP_CELL
        name = ' '.join(person['name'].split()[:2])[:18]

        # Alternate row tint for readability
        if i % 2 == 0:
            dpg.draw_rectangle([0,y1],[TW+20,y1+PP_CELL],
                               fill=(20,22,34,255), color=(0,0,0,0), parent=dl)

        dpg.draw_text([4, y1 + PP_CELL//2 - 7],
                      name, color=LCOL, size=PP_SZ, parent=dl)

        for k in range(M):
            proj_key = PROJECTS[k][0]
            pct      = person['allocs'].get(proj_key, 0)
            x1       = PP_LHDR + k*PP_CELL
            fill     = people_col(pct, CAT[k]) if pct > 0 else ZERO_C
            cat_col  = CAT_C.get(CAT[k], GRID)  # border = category colour
            border   = cat_col if pct > 0 else GRID
            dpg.draw_rectangle([x1,y1],[x1+PP_CELL,y1+PP_CELL],
                               fill=fill, color=border, thickness=1, parent=dl)
            if pct > 0:
                dpg.draw_text([x1+3, y1 + PP_CELL//2 - 6],
                              str(pct), color=(255,255,255,230),
                              size=PP_VAL, parent=dl)


def _draw_people():
    _draw_people_header()
    _draw_people_body()


# ═══════════════════════════════════════════════════════════
# CELL EDITOR POPUP
# ═══════════════════════════════════════════════════════════
_edit = {'type': None, 'j': 0, 'k': 0}

def _show_edit(etype, j, k, cur):
    _edit.update({'type': etype, 'j': j, 'k': k})
    lbl  = 'PHI' if etype == 'phi' else 'ALPHA'
    rng  = '0.0 - 1.0' if etype == 'phi' else '0.0 - 2.0'
    dpg.set_value('edit_title',
                  f"{lbl}  [{SHORT[j]}  <--  {SHORT[k]}]   range {rng}")
    dpg.set_value('edit_val', round(float(cur), 3))
    dpg.configure_item('edit_popup', show=True)

def _apply_edit():
    v = float(dpg.get_value('edit_val'))
    j, k = _edit['j'], _edit['k']
    with lock:
        if _edit['type'] == 'phi':
            if eng: eng.PHI[j,k]   = np.clip(v, 0.0, 1.0)
            snap['PHI'][j,k]       = np.clip(v, 0.0, 1.0)
        else:
            if eng: eng.ALPHA[j,k] = np.clip(v, 0.0, 2.0)
            snap['ALPHA'][j,k]     = np.clip(v, 0.0, 2.0)
    dpg.configure_item('edit_popup', show=False)
    _draw_phi_alpha()   # redraw immediately

def _check_phi_click():
    if not dpg.does_item_exist('dl_phi'): return
    if not dpg.is_item_hovered('dl_phi'): return
    if not dpg.is_mouse_button_clicked(0): return
    rect = dpg.get_item_rect_min('dl_phi')
    mx, my = dpg.get_mouse_pos()
    lx = mx - rect[0]
    ly = my - rect[1]
    k  = int((lx - PM_LHDR) / PM_CELL)
    j  = int((ly - PM_THDR) / PM_CELL)
    if 0 <= j < M and 0 <= k < M and j != k:
        s = snap
        if j < k:
            _show_edit('phi',   j, k, s['PHI'][j,k])
        else:
            _show_edit('alpha', j, k, s['ALPHA'][j,k])


# ═══════════════════════════════════════════════════════════
# MAIN SIMULATION WINDOW
# ═══════════════════════════════════════════════════════════
def build_main_window(W_WIN=1440, H_WIN=920):
    BAR_H  = 46
    LEFT_W = 200
    BODY_H = H_WIN - BAR_H - 10

    with dpg.window(tag='main_win', no_title_bar=True, no_resize=False,
                    no_move=True, no_scrollbar=True,
                    width=W_WIN, height=H_WIN, pos=(0,0), show=False):

        # ── Top command bar ───────────────────────────────────
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=10)
            dpg.add_text("DRVE", color=ACCENT)
            dpg.add_spacer(width=20)
            dpg.add_text("t = 0", tag='t_txt', color=LCOL)
            dpg.add_spacer(width=20)
            dpg.add_text("Speed", color=DIM)
            dpg.add_slider_int(tag='tps_sl', default_value=5,
                               min_value=1, max_value=30, width=110, no_input=True,
                               callback=lambda s,v: ctrl.update({'tps': v}))
            dpg.add_spacer(width=16)
            dpg.add_button(label=" Run  ", tag='btn_run', callback=_toggle_pause)
            dpg.add_spacer(width=6)
            dpg.add_button(label=" Reset ", callback=_do_reset)
            dpg.add_spacer(width=6)
            dpg.add_button(label=" Config ", callback=_go_config)
            dpg.add_spacer(width=20)
            dpg.add_text("[PAUSED]", tag='status_lbl', color=PAU_C)

        dpg.add_separator()

        # ── Body: left panel + tabs ───────────────────────────
        with dpg.group(horizontal=True):

            # Left: W resistance sliders only
            with dpg.child_window(tag='left_win', width=LEFT_W,
                                  height=BODY_H, border=True):
                dpg.add_text("RESISTANCE  W", color=DIM)
                dpg.add_text("drag to change friction", color=DIM)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                for j in range(M):
                    col = CAT_C.get(CAT[j], (148,163,184,200))
                    dpg.add_text(SHORT[j], color=col)
                    dpg.add_slider_float(
                        tag=f'ws_{j}', label=f'##{j}',
                        default_value=2.0,
                        min_value=0.05, max_value=5.0,
                        width=LEFT_W - 20, no_input=True,
                        format="W = %.2f",
                        callback=_make_w_cb(j),
                    )
                    dpg.add_spacer(height=4)

            dpg.add_spacer(width=6)

            # Right: tabs for the two matrices
            right_w = W_WIN - LEFT_W - 20
            with dpg.child_window(width=right_w, height=BODY_H,
                                  border=False, tag='right_panel'):
                with dpg.tab_bar(tag='tabs'):

                    # Tab 1 — PHI / ALPHA dependency matrix
                    with dpg.tab(label="  PHI / ALPHA  Dependencies  ", tag='tab_phi'):
                        dpg.add_text(
                            "Upper triangle (indigo) = PHI gate  |  "
                            "Lower triangle (green) = ALPHA boost  |  "
                            "Click any cell to edit",
                            color=DIM)
                        dpg.add_spacer(height=4)
                        with dpg.child_window(border=False,
                                              horizontal_scrollbar=True,
                                              height=BODY_H - 40,
                                              tag='phi_scroll'):
                            dpg.add_drawlist(
                                tag='dl_phi',
                                width=PM_LHDR + M*PM_CELL + 20,
                                height=PM_THDR + M*PM_CELL + 50,
                            )

                    # Tab 2 — People x Projects allocation
                    with dpg.tab(label="  Team Allocation  ", tag='tab_ppl'):
                        dpg.add_text(
                            "Each row = one engineer   |   "
                            "Fill = time investment (blue low, teal mid, orange high)   |   "
                            "Border = project category",
                            color=DIM)
                        dpg.add_spacer(height=4)

                        # ── Frozen column-header strip ────────────
                        # No scrollbar, x-scroll driven programmatically
                        # from the body scroll below.
                        HDR_H = PM_THDR + 4
                        with dpg.child_window(
                                tag='ppl_hdr_win',
                                width=right_w - 16,
                                height=HDR_H,
                                border=False,
                                no_scrollbar=True):
                            dpg.add_drawlist(
                                tag='dl_ppl_hdr',
                                width=PP_LHDR + M*PP_CELL + 20,
                                height=HDR_H,
                            )

                        # ── Scrollable body (person rows) ─────────
                        with dpg.child_window(
                                tag='ppl_scroll',
                                width=right_w - 16,
                                height=BODY_H - 40 - HDR_H - 4,
                                border=False,
                                horizontal_scrollbar=True):
                            dpg.add_drawlist(
                                tag='dl_ppl',
                                width=PP_LHDR + M*PP_CELL + 20,
                                height=N*PP_CELL + 10,
                            )

    # Edit popup
    with dpg.window(tag='edit_popup', label="Edit Parameter",
                    show=False, modal=True, no_resize=True,
                    width=400, height=140, pos=[520, 400]):
        dpg.add_text("", tag='edit_title', color=LCOL)
        dpg.add_spacer(height=8)
        dpg.add_input_float(tag='edit_val', label="value",
                            default_value=0.0,
                            min_value=0.0, max_value=2.0,
                            step=0.05, width=140)
        dpg.add_spacer(height=10)
        with dpg.group(horizontal=True):
            dpg.add_button(label="  Apply  ", width=90, callback=_apply_edit)
            dpg.add_spacer(width=8)
            dpg.add_button(label=" Set to 0 ", width=90,
                           callback=lambda: (dpg.set_value('edit_val',0.0), _apply_edit()))
            dpg.add_spacer(width=8)
            dpg.add_button(label="  Cancel  ", width=90,
                           callback=lambda: dpg.configure_item('edit_popup', show=False))


# ═══════════════════════════════════════════════════════════
# CONTROLS
# ═══════════════════════════════════════════════════════════
def _make_w_cb(j):
    def cb(s, v):
        with lock:
            if eng: eng.W[j] = float(v)
    return cb

def _toggle_pause():
    ctrl['paused'] = not ctrl['paused']
    if ctrl['paused']:
        dpg.configure_item('btn_run', label=" Run  ")
        dpg.set_value('status_lbl', "[PAUSED]")
        dpg.configure_item('status_lbl', color=PAU_C)
    else:
        dpg.configure_item('btn_run', label=" Pause")
        dpg.set_value('status_lbl', "[RUNNING]")
        dpg.configure_item('status_lbl', color=RUN_C)

def _do_reset():
    global snap
    W_now  = [dpg.get_value(f'ws_{j}') if dpg.does_item_exist(f'ws_{j}') else 2.0
              for j in range(M)]
    g_mult = dpg.get_value('cfg_gmult') if dpg.does_item_exist('cfg_gmult') else 50.0
    was_paused = ctrl['paused']
    ctrl['paused'] = True
    with lock:
        init_engine_with_config(W_now, g_mult, ctrl['tps'], ctrl['dt'])
        snap = {
            't': 0.0, 'W': np.array(W_now),
            'PHI': np.zeros((M,M)), 'ALPHA': np.zeros((M,M)),
            'V': np.zeros(N),
        }
    if dpg.does_item_exist('t_txt'):
        dpg.set_value('t_txt', "t = 0")
    if not was_paused:
        # was running — keep paused after reset, user must press Run again
        dpg.configure_item('btn_run', label=" Run  ")
        dpg.set_value('status_lbl', "[PAUSED]")
        dpg.configure_item('status_lbl', color=PAU_C)
    _draw_phi_alpha()

def _go_config():
    ctrl['paused'] = True
    dpg.configure_item('btn_run', label=" Run  ")
    dpg.set_value('status_lbl', "[PAUSED]")
    dpg.configure_item('status_lbl', color=PAU_C)
    dpg.configure_item('main_win', show=False)
    dpg.configure_item('cfg_win',  show=True)
    dpg.set_primary_window('cfg_win', True)


# ═══════════════════════════════════════════════════════════
# PER-FRAME UPDATE
# ═══════════════════════════════════════════════════════════
_frame = [0]

def _update():
    _frame[0] += 1
    if not dpg.is_item_shown('main_win'): return

    # Sync t counter only when running
    if not ctrl['paused']:
        dpg.set_value('t_txt', f"t = {snap['t']:.0f}")

    # Sync W sliders from engine (in case engine modified W)
    if _frame[0] % 10 == 0 and eng is not None:
        W = snap['W']
        for j in range(M):
            if not dpg.is_item_active(f'ws_{j}'):
                dpg.set_value(f'ws_{j}', float(W[j]))

    # Sync frozen header x-scroll to body x-scroll
    if (dpg.does_item_exist('ppl_scroll') and
            dpg.does_item_exist('ppl_hdr_win')):
        sx = dpg.get_x_scroll('ppl_scroll')
        dpg.set_x_scroll('ppl_hdr_win', sx)

    # Redraw matrices
    if _frame[0] % 4 == 0:
        _draw_phi_alpha()
    if _frame[0] % 12 == 0:
        _draw_people_body()   # body redraws every 12 frames
    if _frame[0] == 3:
        _draw_people_header() # header only needs one draw (static data)

    _check_phi_click()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
FONT_UI   = None   # Segoe UI 15 — UI labels, sliders, buttons
FONT_MONO = None   # Consolas 13 — matrix cell values

def main(skip_config=False):
    global FONT_UI, FONT_MONO
    _, _, W_def, _, _, _, _, _, _, _ = build_tensors()

    dpg.create_context()

    # Load fonts
    with dpg.font_registry():
        try:
            FONT_UI = dpg.add_font("C:/Windows/Fonts/segoeui.ttf", 15)
        except Exception:
            FONT_UI = None
        try:
            FONT_MONO = dpg.add_font("C:/Windows/Fonts/consola.ttf", 13)
        except Exception:
            FONT_MONO = None

    with dpg.theme() as theme:
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,        ( 14, 15, 22))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,         ( 20, 22, 33))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,         ( 30, 32, 48))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered,  ( 42, 45, 65))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,      ( 99,102,241))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive,(119,122,255))
            dpg.add_theme_color(dpg.mvThemeCol_Button,          ( 35, 38, 58))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,   ( 55, 58, 85))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,    ( 99,102,241))
            dpg.add_theme_color(dpg.mvThemeCol_Tab,             ( 22, 24, 36))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered,      ( 50, 54, 80))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive,       ( 38, 42, 62))
            dpg.add_theme_color(dpg.mvThemeCol_Separator,       ( 40, 44, 64))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,         ( 14, 15, 22))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,   ( 22, 24, 36))
            dpg.add_theme_color(dpg.mvThemeCol_Text,            (180,188,210))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,    ( 70, 78,105))
            dpg.add_theme_color(dpg.mvThemeCol_Border,          ( 40, 44, 64))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg,         ( 20, 22, 34))
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg,( 0,  0,  0,160))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,     ( 14, 15, 22))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,   ( 55, 60, 88))
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,   4)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,    3)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,     3)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding,      4)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,     6, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,      8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,   10, 8)
    dpg.bind_theme(theme)
    if FONT_UI:
        dpg.bind_font(FONT_UI)

    with dpg.theme() as launch_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        ( 40, 45,110))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, ( 65, 72,160))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  ( 99,102,241))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,  6)

    build_config_window(W_def)
    dpg.bind_item_theme('btn_launch', launch_theme)
    build_main_window()

    dpg.create_viewport(
        title="DRVE  Dynamic Resource Value Engine",
        width=1440, height=920,
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()

    if skip_config:
        do_launch()
    else:
        dpg.set_primary_window('cfg_win', True)

    while dpg.is_dearpygui_running():
        _update()
        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == '__main__':
    main(skip_config='--skip-config' in sys.argv)
