"""Build synthetic navigation walks from REAL ankle recordings, to test the navigator
pipeline end to end without walking around campus:

    python training/nav_sim.py            # writes data_nav_synthetic/*.json
    python training/nav_eval.py --walks data_nav_synthetic

Real walking/standing samples from data/ are stitched into routes. Those recordings
were made walking in circles, so the rotation about gravity is first projected out of
the gyro, then known turns (and a constant gyro bias, to exercise the calibration) are
injected. Everything after that is the real board code: neural step counter, heading,
turn detector, swing-sized steps, auto-labeller, policy. Two simulated walkers with
different step lengths and cadences walk each route. Only walker A's first walk of each
route has its distance typed in, so B's calibration has to come from the route length.

It tests the software. It can't tell you how well real turns, real stairs or a real
walker's drift behave: only walks on campus can.
"""
import argparse
import csv
import glob
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app" / "python"))
import models    # noqa: E402
import nav       # noqa: E402
import stepnet   # noqa: E402

FS = nav.FS
STEP_MODEL = ROOT / "app" / "python" / "model_steps.json"

# Routes: ("walk", metres) | ("turn", degrees, + = left) | ("stairs_up"/"stairs_down", metres) | ("pause", s)
ROUTES = {
    "Halligan to Tisch": [("walk", 25), ("turn", 90), ("walk", 40), ("turn", -90), ("walk", 15),
                          ("stairs_up", 6), ("walk", 12), ("turn", 90), ("walk", 10)],
    "Tisch to Campus Center": [("walk", 18), ("turn", -90), ("walk", 30), ("stairs_down", 6),
                               ("walk", 8), ("turn", -90), ("walk", 22), ("turn", 90), ("walk", 6)],
}
WALKERS = {  # true step length (m), cadence stretch
    "A": {"true": 0.74, "speed": 1.0},
    "B": {"true": 0.66, "speed": 0.9},
}


def load(pattern):
    out = []
    for p in sorted(glob.glob(str(ROOT / pattern))):
        rows = [[float(v) for v in r[1:7]] for r in list(csv.reader(open(p)))[1:]]
        out.append(rows[5 * FS:-5 * FS])     # README: first/last 5 s are fumbling
    return out


def remove_yaw(rows):
    """Project the rotation about gravity out of the gyro (the recordings circle)."""
    up, a, out = list(rows[0][:3]), 1.0 / (2 * FS), []
    for r in rows:
        up = [u + a * (x - u) for u, x in zip(up, r[:3])]
        n = math.sqrt(sum(u * u for u in up))
        uh = [u / n for u in up]
        w = sum(g * u for g, u in zip(r[3:], uh))
        out.append(r[:3] + [g - w * u for g, u in zip(r[3:], uh)] + uh)
    return out          # ax ay az gx gy gz ux uy uz


def stretch(rows, f):
    """Play a recording f times as fast (linear interpolation)."""
    out, t = [], 0.0
    while t < len(rows) - 1:
        i = int(t)
        fr = t - i
        out.append([a + fr * (b - a) for a, b in zip(rows[i], rows[i + 1])])
        t += f
    return out


class Source:
    """Endless stream of de-yawed samples from a set of recordings."""

    def __init__(self, recs, rng):
        self.recs, self.rng = recs, rng
        self.cur, self.i = rng.choice(recs), 0

    def take(self, n):
        out = []
        while len(out) < n:
            if self.i >= len(self.cur):
                self.cur, self.i = self.rng.choice(self.recs), 0
            out.append(self.cur[self.i])
            self.i += 1
        return out


def simulate(route, walker, rng, walking, standing, cadence):
    """-> (samples, marks, true metres walked). Leg lengths vary +-4%, turns +-10 deg."""
    w = WALKERS[walker]
    walk_src = Source([stretch(r, w["speed"]) for r in walking], rng)
    stand_src = Source(standing, rng)
    bias = [rng.uniform(-6, 6) for _ in range(3)]
    steps_per_s = cadence * w["speed"]
    samples, marks, yaw, metres_total = [], [], [], 0.0

    def add(rows, yaw_rate=None):
        for k, r in enumerate(rows):
            samples.append(r)
            yaw.append(yaw_rate[k] if yaw_rate else 0.0)

    add(stand_src.take(3 * FS))                                    # stand still: calibration
    for leg in ROUTES[route]:
        kind = leg[0]
        if kind == "turn":
            ang = leg[1] + rng.uniform(-10, 10)
            n = int(rng.uniform(1.6, 2.4) * FS)
            prof = [math.sin(math.pi * (k + 0.5) / n) ** 2 for k in range(n)]
            s = sum(prof) / FS
            add(walk_src.take(n), [ang * p / s for p in prof])
            metres_total += n / FS * steps_per_s * w["true"]
            continue
        if kind == "pause":
            add(stand_src.take(int(leg[1] * FS)))
            continue
        metres = leg[1] * rng.uniform(0.96, 1.04)
        n = int(metres / w["true"] / steps_per_s * FS)
        metres_total += metres
        if kind.startswith("stairs"):
            marks.append({"t": round((len(samples) + rng.randint(-25, 50)) / FS, 2), "kind": kind})
            add(walk_src.take(n))
            marks.append({"t": round((len(samples) + rng.randint(0, 50)) / FS, 2), "kind": "stairs_end"})
        else:
            add(walk_src.take(n))
    add(stand_src.take(4 * FS))                                    # arrived, standing
    out = []
    for r, yr in zip(samples, yaw):
        uh = r[6:9]
        g = [gi + yr * u + b + rng.gauss(0, 0.3) for gi, u, b in zip(r[3:6], uh, bias)]
        out.append(r[:3] + g)
    return out, marks, metres_total


def run_odometry(samples, counter, classify):
    odo = nav.Odometry(counter, classify=classify)
    rows = []
    for s in samples:
        row = odo.push(s)
        if row:
            rows.append([row[c] if c in row else -1 for c in nav.TRACE_COLS])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data_nav_synthetic"))
    ap.add_argument("--per-walker", type=int, default=2, help="walks per walker per route (10 max total)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    walking = [remove_yaw(r) for r in load("data/walking/*.csv")]
    standing = [remove_yaw(r) for r in load("data/standing/*.csv")]
    counter = stepnet.load(STEP_MODEL)

    # Cadence of the real walking data, from the step counter itself.
    n = 0
    for r in walking:
        counter.reset()
        for s in r:
            counter.push(s[:6])
        n += counter.total
    cadence = n / (sum(len(r) for r in walking) / FS)
    print(f"real walking cadence: {cadence:.2f} steps/s")

    classify = models.load(ROOT / "app" / "python" / "model_full.json").predict
    store = nav.WalkStore(args.out)
    for wid in store.ids():
        store.delete(wid)
    book = nav.RouteBook(args.out)
    for name in list(book.routes):
        book.delete(name)
    for name in ROUTES:
        book.create(name)          # lengths unknown: only walker A's typed distances calibrate
    k, walks = 0, []
    for route in ROUTES:
        for rep in range(args.per_walker):
            for walker in WALKERS:
                samples, marks, true_m = simulate(route, walker, rng, walking, standing, cadence)
                counter = stepnet.load(STEP_MODEL)       # fresh history, like a new session on the board
                typed = round(true_m) if (walker == "A" and rep == 0) else None
                walk = nav.new_walk("demo", route, walker, typed)
                walk["id"] = f"sim{k:02d}_{walker}_{route.split()[0].lower()}"
                walk["marks"] = marks
                walk["true_m"] = round(true_m, 1)
                walk["trace"]["rows"] = run_odometry(samples, counter, classify)
                walks.append(walk)
                k += 1
    cal = nav.calibrate(walks)
    print("calibration:", {w: (round(c["k"] * nav.DEFAULT_SWING_PER_STEP, 3), c["source"]) for w, c in cal.items()},
          "(m per 60 deg of swing)")
    for walk in walks:
        kk = nav.k_for(cal, walk["walker"])
        walk["labels"], turns = nav.auto_label(walk, kk)
        walk["reviewed"] = walk["include"] = True
        store.save(walk)
        s = nav.WalkStore.summary(walk, kk)
        print(f"{walk['id']:<24} {s['seconds']:4d} s  est {s['dist']:4d} m  true {walk['true_m']:5.0f} m"
              f"{'  (typed)' if walk['distance_m'] else '':9}  turns {[round(t['angle']) for t in turns]}")



if __name__ == "__main__":
    main()
