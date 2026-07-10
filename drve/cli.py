"""
cli.py — Run the DRVE simulation from the terminal.

Usage:
  python cli.py                        # run with defaults, live table output
  python cli.py --ticks 500            # stop after 500 ticks
  python cli.py --tps 0                # run as fast as possible (no sleep)
  python cli.py --W 6=3.5             # override W[6] (CyRo resistance = 3.5)
  python cli.py --W 6=3.5 --W 8=2.0  # multiple overrides
  python cli.py --reset --ticks 200    # fresh sim, 200 ticks, then exit
  python cli.py --csv out.csv          # write tick-by-tick S[] to CSV
  python cli.py --watch 1,6,8          # show only projects 1, 6, 8 (by index)
  python cli.py --quiet                # no table, just final summary
"""

import argparse, sys, time, os, csv as csv_mod
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data              import build_tensors, PROJECTS
from simgraph_adapter  import SimGraphAdapter


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='DRVE CLI runner')
    p.add_argument('--ticks',  type=int,   default=0,     help='stop after N ticks (0 = run forever)')
    p.add_argument('--tps',    type=float, default=10.0,  help='ticks per second (0 = max speed)')
    p.add_argument('--dt',     type=float, default=1.0,   help='dt per tick')
    p.add_argument('--W',      action='append', default=[], metavar='J=V',
                               help='override W[j], e.g. --W 6=2.5')
    p.add_argument('--G',      action='append', default=[], metavar='J=V',
                               help='override G[j], e.g. --G 8=50000')
    p.add_argument('--phi',    action='append', default=[], metavar='J,K=V',
                               help='override PHI[j,k], e.g. --phi 0,6=0.9')
    p.add_argument('--alpha',  action='append', default=[], metavar='J,K=V',
                               help='override ALPHA[j,k], e.g. --alpha 1,10=0.5')
    p.add_argument('--watch',  type=str,   default='',    help='comma-separated project indices to show')
    p.add_argument('--csv',    type=str,   default='',    help='path to write CSV output')
    p.add_argument('--quiet',  action='store_true',       help='suppress live table, only show final')
    p.add_argument('--interval', type=int, default=10,   help='print table every N ticks')
    return p.parse_args()


# ── Build engine ──────────────────────────────────────────────────────────────

def build_engine(args):
    P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys = build_tensors()

    # Bootstrap G from one DRVEngine tick
    from engine import DRVEngine
    boot = DRVEngine(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)
    s0 = boot.tick()
    G  = np.maximum(s0['flow'] * 50.0, 1.0)

    eng = SimGraphAdapter(P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys)

    # Apply CLI overrides
    for spec in args.W:
        j, v = spec.split('='); eng.W[int(j)] = float(v)
        print(f'  W[{j}] = {v}  ({proj_keys[int(j)]})')

    for spec in args.G:
        j, v = spec.split('='); eng.G[int(j)] = float(v)
        print(f'  G[{j}] = {v}  ({proj_keys[int(j)]})')

    for spec in args.phi:
        jk, v = spec.split('='); j, k = jk.split(',')
        eng.PHI[int(j), int(k)] = float(v)
        print(f'  PHI[{j},{k}] = {v}')

    for spec in args.alpha:
        jk, v = spec.split('='); j, k = jk.split(',')
        eng.ALPHA[int(j), int(k)] = float(v)
        print(f'  ALPHA[{j},{k}] = {v}')

    return eng, proj_keys, names


# ── Table rendering ───────────────────────────────────────────────────────────

COLS = 80

def clear_lines(n):
    for _ in range(n):
        sys.stdout.write('\x1b[1A\x1b[2K')

def bar(s, width=12):
    filled = int(s * width)
    return '[' + '#' * filled + '.' * (width - filled) + ']'

def print_table(t, eng, proj_keys, watch_idx, dt_ms):
    S    = eng.get_S()
    R    = eng.readiness()
    B    = eng.boost()
    flow = eng.contribution().sum(axis=0)

    indices = watch_idx if watch_idx else list(range(len(proj_keys)))

    print(f'\n  t={t:.0f}  dt={dt_ms:.1f}ms/tick', flush=True)
    print(f'  {"#":>3}  {"Project":<28}  {"S":>5}  {"bar":<14}  {"flow":>6}  {"R":>5}  {"B":>5}')
    print('  ' + '-' * 68)

    for j in indices:
        s = min(S[j], 1.0)
        flag = ' DONE' if s >= 0.999 else (' BLKD' if R[j] < 0.5 else '')
        print(f'  {j:>3}  {proj_keys[j][:28]:<28}  {s:>4.0%}  {bar(s):<14}  '
              f'{flow[j]:>6.0f}  {R[j]:>5.2f}  {B[j]:>5.2f}{flag}')

    done = sum(1 for s in S if s >= 0.999)
    print(f'\n  {done}/{len(proj_keys)} complete', flush=True)
    return 4 + len(indices)   # lines printed (for cursor rewind)


# ── CSV writer ────────────────────────────────────────────────────────────────

def open_csv(path, proj_keys):
    f = open(path, 'w', newline='')
    w = csv_mod.writer(f)
    w.writerow(['t'] + [f'S_{k}' for k in proj_keys] + [f'flow_{k}' for k in proj_keys])
    return f, w

def write_csv_row(w, t, eng, proj_keys):
    S    = eng.get_S().tolist()
    flow = eng.contribution().sum(axis=0).tolist()
    w.writerow([round(t, 1)] + [round(s, 4) for s in S] + [round(f, 2) for f in flow])


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print('\nDRVE CLI')
    print('=' * 40)

    eng, proj_keys, names = build_engine(args)

    watch_idx = [int(x) for x in args.watch.split(',') if x.strip()] if args.watch else []

    csv_file = csv_writer = None
    if args.csv:
        csv_file, csv_writer = open_csv(args.csv, proj_keys)
        print(f'  Writing CSV → {args.csv}')

    sleep_s  = (1.0 / args.tps) if args.tps > 0 else 0
    max_tick = args.ticks
    interval = args.interval
    dt       = args.dt

    print(f'  tps={args.tps or "max"}  dt={dt}  ticks={max_tick or "inf"}')
    print()

    last_lines = 0
    tick       = 0

    try:
        while True:
            t0 = time.perf_counter()
            eng.tick(dt)
            tick += 1
            dt_ms = (time.perf_counter() - t0) * 1000

            if csv_writer:
                write_csv_row(csv_writer, eng.t, eng, proj_keys)

            if not args.quiet and tick % interval == 0:
                if last_lines:
                    clear_lines(last_lines)
                last_lines = print_table(eng.t, eng, proj_keys, watch_idx, dt_ms)

            if max_tick and tick >= max_tick:
                break

            if sleep_s:
                elapsed = time.perf_counter() - t0
                rem = sleep_s - elapsed
                if rem > 0:
                    time.sleep(rem)

    except KeyboardInterrupt:
        print('\n\n  Stopped.')

    # Final summary
    S    = eng.get_S()
    flow = eng.contribution().sum(axis=0)
    print(f'\n  Final state at t={eng.t:.0f}')
    print('  ' + '-' * 55)
    for j, k in enumerate(proj_keys):
        s = min(S[j], 1.0)
        status = 'DONE' if s >= 0.999 else f'{s:.0%}'
        print(f'  {j:>2}  {k[:38]:<38}  {status}')

    if csv_file:
        csv_file.close()
        print(f'\n  CSV saved → {args.csv}')

    print()


if __name__ == '__main__':
    main()
