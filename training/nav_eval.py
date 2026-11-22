"""Leave-one-walk-out evaluation of the navigation policy, on the laptop.

    scp -r arduino@<board>.local:ArduinoApps/<app>/walks .     # or adb pull
    python training/nav_eval.py --walks walks

Each route is evaluated on its own, with its own turn threshold and length from the
board's walks/routes.json. Uses each walk's saved labels (review them on the board's Label page first). Only walks
ticked 'use for training' are used, the same set the board trains on. Each walk is
replayed through a policy trained on the other walks of its route, with the same
smoothing as live navigation. Reports, per walk: how often the instruction matched the
label, how many maneuvers (turns, stairs, arrival) were announced and how early, and
false alarms (an instruction where the label had none).

Distance check: for every walk whose distance you typed in, the distance error when that
walker's calibration comes only from the OTHER walks, for swing-sized steps (what the
board uses) and for a fixed step length (steps x one number). If swing doesn't beat the
fixed length on your walks, it's not helping. Also writes training/report_nav.txt.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app" / "python"))
import nav  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks", default=str(ROOT / "walks"))
    ap.add_argument("--route", default=None, help="only this route (default: every route)")
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--turn-min", type=float, default=None,
                    help="turn threshold (deg) for every route; default: each route's own setting")
    ap.add_argument("--all", action="store_true", help="include walks not ticked for training")
    args = ap.parse_args()
    walks = [w for w in nav.WalkStore(args.walks).all() if args.all or w.get("include")]
    if not walks:
        sys.exit(f"no training walks in {args.walks}")
    book = nav.RouteBook(args.walks)
    lengths = book.lengths()
    cal = nav.calibrate(walks, route_lengths=lengths)
    routes = [args.route] if args.route else sorted({w["route"] for w in walks}, key=str.lower)
    lines = ["stride calibration (all routes): " + ", ".join(
        f"{wk} {c['k'] * nav.DEFAULT_SWING_PER_STEP:.3f} m per 60 deg swing ({c['source']})"
        for wk, c in cal.items())]
    tot = {"hit": 0, "man": 0, "fa": 0}
    dist = []
    for route in routes:
        mine = [w for w in walks if w["route"] == route]
        tm = args.turn_min if args.turn_min is not None else book.turn_min(route)
        lines += ["", f"=== {route}: {len(mine)} walks ({sum(w['kind'] == 'dagger' for w in mine)} DAgger), "
                      f"limit {nav.MAX_TRAIN_WALKS}, turns >= {tm:.0f} deg, "
                      f"length {str(lengths[route]) + ' m' if route in lengths else 'unknown'}",
                  f"{'walk':<28}{'walker':<10}{'acc':>6}  maneuvers  false alarms  announced early by (s)"]
        for r in nav.evaluate(walks, k=args.k, turn_min=tm, only_route=route, route_lengths=lengths):
            if "dist_err_swing" in r:
                dist.append(r)
            if "skipped" in r:
                lines.append(f"{r['id']:<28}{r['walker']:<10}  skipped: {r['skipped']}")
                continue
            lines.append(f"{r['id']:<28}{r['walker'][:9]:<10}{r['accuracy']:>6.0%}  "
                         f"{r['hit']:>3}/{r['maneuvers']:<3}     {r['false_alarms']:>3}         {r['lead_s']}")
            tot["hit"] += r["hit"]; tot["man"] += r["maneuvers"]; tot["fa"] += r["false_alarms"]
    lines += ["", f"maneuvers announced: {tot['hit']}/{tot['man']}   false alarms: {tot['fa']}"]
    lines += ["", "distance check (calibrated on the other walks; error vs the distance you typed):"]
    if not dist:
        lines.append("  no walks with a typed distance")
    for r in dist:
        lines.append(f"  {r['id']:<28}swing-sized steps {r['dist_err_swing']:+6.1%}   "
                     f"fixed step length {r['dist_err_steps']:+6.1%}")
    if dist:
        mae = lambda key: sum(abs(r[key]) for r in dist) / len(dist)
        lines.append(f"  mean |error|: swing {mae('dist_err_swing'):.1%}, fixed step {mae('dist_err_steps'):.1%}")
    text = "\n".join(lines)
    print(text)
    (ROOT / "training" / "report_nav.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
