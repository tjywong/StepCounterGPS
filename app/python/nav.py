"""Campus navigator: dead reckoning on top of the step counter, plus a behaviour-cloned
policy that says what to do next (straight / left / right / stairs / arrived).

Pure Python, shared by the board (main.py) and the laptop (training/nav_*.py), so the
features the policy is trained on are computed by exactly the same code it runs with.

Pipeline, one IMU sample at a time:

    IMU 50 Hz --> StepCounter (neural net) ----------> steps
              |-> Heading: gyro about gravity -------> heading
              |            gyro across gravity ------> shank swing per step
              |   distance = sum over steps of (swing of that step) x k   (k per walker)
              `-> activity classifier (random forest) -> stairs up / down / walking ...
    every 0.2 s: one trace row (t, steps, swing, heading, activity)
    TurnDetector(heading, threshold) -> turns so far, swing since last turn
    features -> Policy (k-nearest demonstration moments of this route) -> instruction

No GPS, wifi or magnetometer: the MPU6050 has none. Heading is the integral of the gyro
component along gravity, so it doesn't matter how the strap is rotated on the ankle.

Distance: a longer step swings the shank through a bigger angle (pendulum: step length ~
leg length x swing angle). So each counted step is sized by the shank's rotation across
gravity during it, and one number per walker, k (metres per degree of swing), turns that
into metres. k is fitted from walks where you typed the distance you actually walked.
"""
import bisect
import collections
import hashlib
import json
import math
import os
import re
import time

FS = 50                  # IMU rate (Hz), set by the sketch
TRACE_EVERY = 10         # samples per trace row -> 5 Hz
CALIB_SAMPLES = 100      # stand still this long (2 s) at the start to measure gyro bias
CALIB_MAX_DPS = 12.0     # gyro wobble allowed while calibrating
UP_TAU_S = 2.0           # low-pass for the gravity direction
STILL_DPS = 3.0          # below this (after bias) and ~1 g: the foot is still
STILL_SAMPLES = 25       # 0.5 s still -> stop integrating heading, refine bias

SWING_TAU_S = 2.0        # swing per step = (swing in last ~2 s) / (steps in last ~2 s)
MIN_CADENCE = 0.5        # steps/s; below this keep the last swing-per-step
SWING_RANGE = (20.0, 150.0)
DEFAULT_SWING_PER_STEP = 60.0           # deg, measured on the walking recordings in data/
K_DEFAULT = 0.70 / DEFAULT_SWING_PER_STEP   # m per deg of swing: a 0.70 m step at 60 deg
STEP_DEFAULT = 0.70      # m per step, for the step-count-only baseline in evaluate()

TURN_MIN_DEFAULT = 45.0  # smaller heading changes are wiggles, not turns (a setting)
LEAD_M_DEFAULT = 6.0     # auto-labels announce turns and stairs this far ahead (a setting)
TURN_END_DPS = 10.0      # turn is over when heading settles below this rate
TURN_REF_TAU_S = 3.0     # reference heading follows gentle path curves

ACTIONS = ["straight", "left", "right", "stairs_up", "stairs_down", "arrived"]
STAIRS = ("stairs_up", "stairs_down")
MAX_TRAIN_WALKS = 10     # assignment limit per route: demonstrations + DAgger runs
STAIRS_ONSET_S = 2.5     # the classifier's 2.56 s window + vote reports stairs this late
STAIRS_OFFSET_S = 1.3

TRACE_COLS = ["t", "steps", "swing", "heading", "activity", "calibrating", "policy", "expert"]


# ---------------------------------------------------------------- odometry

class Heading:
    """Integrates the gyro about the (low-passed) gravity axis. Positive = left turn
    (counter-clockwise seen from above), whichever way the sensor is mounted.
    Also reports `perp`: the rotation rate across gravity (deg/s), i.e. the shank swing."""

    def __init__(self):
        self.bias = [0.0, 0.0, 0.0]
        self.up = None
        self.heading = 0.0
        self.perp = 0.0
        self.calib = []
        self.ready = False
        self.still = 0
        self._alpha = 1.0 / (UP_TAU_S * FS)

    def push(self, s):
        a, g = s[:3], s[3:]
        if not self.ready:
            self.calib.append(s)
            if len(self.calib) >= CALIB_SAMPLES:
                gm = [sum(c[3 + i] for c in self.calib) / len(self.calib) for i in range(3)]
                wobble = max(max(abs(c[3 + i] - gm[i]) for i in range(3)) for c in self.calib)
                if wobble > CALIB_MAX_DPS:        # moved: try again with fresh samples
                    self.calib = self.calib[len(self.calib) // 2:]
                    return
                self.bias = gm
                self.up = [sum(c[i] for c in self.calib) / len(self.calib) for i in range(3)]
                self.ready = True
                self.calib = []
            return
        up = self.up
        for i in range(3):
            up[i] += self._alpha * (a[i] - up[i])
        n = math.sqrt(up[0] ** 2 + up[1] ** 2 + up[2] ** 2) or 1.0
        w = [g[i] - self.bias[i] for i in range(3)]
        yaw = (w[0] * up[0] + w[1] * up[1] + w[2] * up[2]) / n
        total2 = w[0] ** 2 + w[1] ** 2 + w[2] ** 2
        self.perp = math.sqrt(max(0.0, total2 - yaw * yaw))
        amag = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
        if math.sqrt(total2) < STILL_DPS and abs(amag - 1.0) < 0.06:
            self.still += 1
        else:
            self.still = 0
        if self.still >= STILL_SAMPLES:
            # Foot is planted: heading can't change. Slowly re-learn the bias (it drifts with temperature).
            for i in range(3):
                self.bias[i] += 0.01 * w[i]
            return
        self.heading += yaw / FS


class TurnDetector:
    """Turns are heading changes of at least `min_deg` that settle again. Runs at the trace
    rate. The reference heading slowly follows gentle path curves, which count as straight."""

    def __init__(self, min_deg=TURN_MIN_DEFAULT, hz=FS / TRACE_EVERY):
        self.hz = hz
        self.min_deg = min_deg
        self.start_deg = min(25.0, 0.6 * min_deg)
        self.ref = None
        self.turning = False
        self.start_ref = 0.0
        self.hist = []                   # (t, heading), last few seconds
        self.n_turns = 0
        self.settled = 0
        self._a = 1.0 / (TURN_REF_TAU_S * hz)

    def step(self, t, h):
        """Returns a turn event {t_start, t_end, angle} when a turn completes, else None."""
        self.hist.append((t, h))
        if len(self.hist) > 8 * self.hz:
            self.hist.pop(0)
        if self.ref is None:
            self.ref = h
            return None
        if not self.turning:
            if abs(h - self.ref) > self.start_deg:
                self.turning, self.start_ref, self.settled = True, self.ref, 0
            else:
                self.ref += self._a * (h - self.ref)
            return None
        back = self.hist[-int(self.hz) - 1] if len(self.hist) > self.hz else self.hist[0]
        rate = abs(h - back[1]) / max(1e-6, t - back[0])
        self.settled = self.settled + 1 if rate < TURN_END_DPS else 0
        if self.settled < self.hz // 2:
            return None
        self.turning = False
        angle = h - self.start_ref
        self.ref = h
        if abs(angle) < self.min_deg:
            return None
        # Turn began where the heading first left the old direction (within 5 deg).
        t_start = self.hist[0][0]
        for tt, hh in reversed(self.hist):
            if abs(hh - self.start_ref) < 5.0:
                t_start = tt
                break
        self.n_turns += 1
        return {"t_start": t_start, "t_end": t, "angle": round(angle, 1)}


class Odometry:
    """Step counter + swing-sized steps + heading + activity + turn detector.
    push() returns a trace row (dict) every 0.2 s, else None.

    `classify(window) -> activity name` is optional (the gait random forest); it runs
    every `hop` samples on the last `window` samples, majority-voted over 3."""

    def __init__(self, step_counter, turn_min=TURN_MIN_DEFAULT, classify=None, window=128, hop=64):
        self.steps = step_counter
        self.steps.reset()
        self.head = Heading()
        self.turns = TurnDetector(turn_min)
        self.n = 0
        self.events = []
        # The step counter reports each step `delay` samples late; delay the swing to match.
        self._swing_hist = collections.deque([0.0], maxlen=getattr(step_counter, "delay", 0) + 1)
        self._a = 1.0 / (SWING_TAU_S * FS)
        self._sw = self._st = 0.0
        self.swing_per_step = DEFAULT_SWING_PER_STEP
        self.swing = 0.0
        self.swing_at_turn = 0.0
        self.classify, self.window, self.hop = classify, window, hop
        self._buf = collections.deque(maxlen=window)
        self._votes = collections.deque(maxlen=3)
        self.activity = ""

    def push(self, sample):
        density = self.steps.push(sample)
        self.head.push(sample)
        self._swing_hist.append(self.head.perp / FS)
        self._sw += self._a * (self._swing_hist[0] - self._sw)
        self._st += self._a * (density - self._st)
        if self._st * FS > MIN_CADENCE:
            self.swing_per_step = min(SWING_RANGE[1], max(SWING_RANGE[0], self._sw / self._st))
        self.swing += density * self.swing_per_step
        self.n += 1
        if self.classify:
            self._buf.append(sample)
            if len(self._buf) == self.window and self.n % self.hop == 0:
                self._votes.append(self.classify(list(self._buf)))
                self.activity = collections.Counter(self._votes).most_common(1)[0][0]
        if self.n % TRACE_EVERY:
            return None
        t = round(self.n / FS, 2)
        calibrating = not self.head.ready
        heading, swing = round(self.head.heading, 1), round(self.swing, 1)
        if not calibrating:
            # Step on the rounded values, exactly as derive() will replay them from the trace.
            ev = self.turns.step(t, heading)
            if ev:
                self.swing_at_turn = swing
                self.events.append(ev)
        return {"t": t, "steps": round(self.steps.total, 2), "swing": swing, "heading": heading,
                "activity": self.activity, "calibrating": int(calibrating),
                "n_turns": self.turns.n_turns, "swing_at_turn": self.swing_at_turn,
                "swing_per_step": round(self.swing_per_step)}


def trace_rows(walk):
    cols = walk["trace"]["cols"]
    return [dict(zip(cols, r)) for r in walk["trace"]["rows"]]


def derive(walk, turn_min=TURN_MIN_DEFAULT):
    """Trace rows plus the turn-detector state the live board would have had with this
    threshold: n_turns and swing_at_turn per row, and the list of turns. Recomputed from
    the stored heading, so changing the threshold re-labels and re-trains consistently."""
    det, at, turns, rows = TurnDetector(turn_min), 0.0, [], trace_rows(walk)
    for r in rows:
        if not r["calibrating"]:
            ev = det.step(r["t"], r["heading"])
            if ev:
                turns.append(ev)
                at = r["swing"]
        r["n_turns"], r["swing_at_turn"] = det.n_turns, at
    return rows, turns


# ---------------------------------------------------------------- features

FEATURES = ["dist", "dist_since_turn", "heading", "n_turns"]
# kNN distance scales: one turn apart (2 units) outweighs everything else, then ~3 m along
# the current leg, ~15 m of total distance (it drifts), 45 deg of heading.
SCALES = [15.0, 3.0, 45.0, 0.5]


def features(row, k):
    """Observation for the policy. Only causal, live quantities: what the board knows now.
    k = metres per degree of swing for this walker."""
    return [row["swing"] * k,
            (row["swing"] - row["swing_at_turn"]) * k,
            row["heading"],
            row["n_turns"]]


# ---------------------------------------------------------------- distance calibration

def _last(walk, col):
    rows = walk["trace"]["rows"]
    return dict(zip(walk["trace"]["cols"], rows[-1]))[col] if rows else 0.0


def calibrate(walks, unit="swing", route_lengths=None):
    """Metres per unit (degree of swing, or step for the baseline) for each walker:
      1. measured: from that walker's walks where they typed the distance walked;
      2. route length: a walk with no distance, on a route someone did measure;
      3. matched: no measurement at all, but a partner walked the same route: assume the
         same distance as the partner's (calibrated) walks of it;
      4. default (a 0.70 m step).
    route_lengths: {route: metres} from the route book, used where no walk of that route
    has a typed distance. Returns {walker: {"k", "source", "walks"}}."""
    default = K_DEFAULT if unit == "swing" else STEP_DEFAULT
    lo, hi = 0.4 * default, 2.5 * default
    route_len = {}
    for w in walks:
        if w.get("distance_m"):
            route_len.setdefault(w["route"], []).append(w["distance_m"])
    route_len = {r: sum(v) / len(v) for r, v in route_len.items()}
    for r, m in (route_lengths or {}).items():
        if m and r not in route_len:
            route_len[r] = m
    out = {}
    for wk in sorted({w["walker"] for w in walks}):
        mine = [w for w in walks if w["walker"] == wk and _last(w, unit) > 0]
        own = [(w["distance_m"], _last(w, unit)) for w in mine if w.get("distance_m")]
        pairs = own or [(route_len[w["route"]], _last(w, unit)) for w in mine if w["route"] in route_len]
        if pairs:
            out[wk] = {"k": min(hi, max(lo, sum(d for d, _ in pairs) / sum(u for _, u in pairs))),
                       "source": "measured" if own else "route length", "walks": len(pairs)}
    for wk in sorted({w["walker"] for w in walks} - set(out)):
        pairs = []
        for w in walks:
            if w["walker"] != wk or _last(w, unit) <= 0:
                continue
            ref = [out[o["walker"]]["k"] * _last(o, unit) for o in walks
                   if o["route"] == w["route"] and o["walker"] in out]
            if ref:
                pairs.append((sum(ref) / len(ref), _last(w, unit)))
        out[wk] = ({"k": min(hi, max(lo, sum(d for d, _ in pairs) / sum(u for _, u in pairs))),
                    "source": "matched to partner", "walks": len(pairs)} if pairs
                   else {"k": default, "source": "default", "walks": 0})
    for v in out.values():
        v["k"] = round(v["k"], 6)
    return out


def k_for(cal, walker, unit="swing"):
    return cal.get(walker, {}).get("k", K_DEFAULT if unit == "swing" else STEP_DEFAULT)


# ---------------------------------------------------------------- labels

def labels_per_row(times, segments):
    """Segments [{t0, t1, action}] -> an action name per trace row (default straight).
    Later segments win where they overlap."""
    out = ["straight"] * len(times)
    for seg in segments:
        lo = bisect.bisect_left(times, seg["t0"])
        hi = bisect.bisect_right(times, seg["t1"])
        for i in range(lo, hi):
            out[i] = seg["action"]
    return out


def segments_from_labels(times, labels):
    segs, i = [], 0
    while i < len(labels):
        j = i
        while j + 1 < len(labels) and labels[j + 1] == labels[i]:
            j += 1
        if labels[i] != "straight":
            segs.append({"t0": times[i], "t1": times[j], "action": labels[i]})
        i = j + 1
    return segs


def _time_at_dist(times, dists, d):
    """First time the walker had covered distance d."""
    i = bisect.bisect_left(dists, d)
    return times[min(i, len(times) - 1)]


def stairs_runs(rows):
    """Stretches the activity classifier called stairs, shifted back by its reporting lag."""
    runs, i = [], 0
    while i < len(rows):
        a = rows[i].get("activity")
        if a in STAIRS:
            j = i
            while j + 1 < len(rows) and rows[j + 1].get("activity") == a:
                j += 1
            runs.append({"t0": max(rows[0]["t"], rows[i]["t"] - STAIRS_ONSET_S),
                         "t1": max(rows[i]["t"], rows[j]["t"] - STAIRS_OFFSET_S), "action": a})
            i = j + 1
        else:
            i += 1
    return runs


def auto_label(walk, k, turn_min=TURN_MIN_DEFAULT, lead_m=LEAD_M_DEFAULT, arrive_m=2.0):
    """Hindsight labels. In a demonstration you already walked the route, so the turns the
    gyro saw ARE the expert's actions: label each one from `lead_m` before it starts until
    it ends. Stairs: your taps, plus flights the classifier found where you didn't tap.
    'arrived' is the last `arrive_m`. Expert corrections from a DAgger run go on last.
    Returns (segments, turns) for the editor to review."""
    rows, turns = derive(walk, turn_min)
    if not rows:
        return [], []
    times = [r["t"] for r in rows]
    dists = [r["swing"] * k for r in rows]
    for i in range(1, len(dists)):
        dists[i] = max(dists[i], dists[i - 1])

    def lead_from(t):
        return _time_at_dist(times, dists, dists[bisect.bisect_left(times, t) if t <= times[-1] else -1] - lead_m)

    segs = []
    for ev in turns:
        segs.append({"t0": lead_from(ev["t_start"]), "t1": ev["t_end"],
                     "action": "left" if ev["angle"] > 0 else "right"})
    marks = sorted(walk.get("marks", []), key=lambda m: m["t"])
    tapped = []
    for i, m in enumerate(marks):
        if m["kind"] not in STAIRS:
            continue
        end = next((n["t"] for n in marks[i + 1:] if n["kind"] == "stairs_end"), m["t"] + 8.0)
        tapped.append({"t0": m["t"], "t1": end, "action": m["kind"]})
    found = [r for r in stairs_runs(rows)
             if not any(r["t0"] <= s["t1"] and s["t0"] <= r["t1"] for s in tapped)]
    for s in tapped + found:
        segs.append({"t0": lead_from(s["t0"]), "t1": s["t1"], "action": s["action"]})
    segs.append({"t0": _time_at_dist(times, dists, dists[-1] - arrive_m), "t1": times[-1],
                 "action": "arrived"})
    segs.extend(walk.get("corrections", []))
    # Normalise through per-row labels so overlaps resolve to clean, non-overlapping segments.
    return segments_from_labels(times, labels_per_row(times, segs)), turns


# ---------------------------------------------------------------- storage

class WalkStore:
    """walks/<id>.json (metadata, marks, labels, 5 Hz odometry trace) + <id>.imu.csv (raw)."""

    RESERVED = ("policy.json", "settings.json", "routes.json")

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def path(self, wid):
        return os.path.join(self.root, wid + ".json")

    def ids(self):
        return sorted(f[:-5] for f in os.listdir(self.root)
                      if f.endswith(".json") and f not in self.RESERVED)

    def load(self, wid):
        with open(self.path(wid)) as f:
            return json.load(f)

    def save(self, walk):
        tmp = self.path(walk["id"]) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(walk, f, separators=(",", ":"))
        os.replace(tmp, self.path(walk["id"]))

    def delete(self, wid):
        for p in (self.path(wid), os.path.join(self.root, wid + ".imu.csv")):
            if os.path.exists(p):
                os.remove(p)

    def all(self):
        out = []
        for wid in self.ids():
            try:
                w = self.load(wid)
                if "swing" in w["trace"]["cols"]:      # skip walks from older formats
                    out.append(w)
            except (OSError, ValueError, KeyError):
                pass
        return out

    @staticmethod
    def summary(w, k):
        last = dict(zip(w["trace"]["cols"], w["trace"]["rows"][-1])) if w["trace"]["rows"] \
            else {"t": 0, "steps": 0, "swing": 0}
        return {"id": w["id"], "kind": w["kind"], "route": w["route"], "walker": w["walker"],
                "seconds": round(last["t"]), "steps": round(last["steps"]),
                "dist": round(last["swing"] * k), "distance_m": w.get("distance_m"),
                "include": w.get("include", False), "reviewed": w.get("reviewed", False),
                "corrections": len(w.get("corrections", []))}


def new_walk(kind, route, walker, distance_m=None):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return {"id": f"{stamp}_{kind}", "kind": kind, "route": route, "walker": walker,
            "distance_m": distance_m, "created": stamp, "include": False,
            "reviewed": False, "marks": [], "corrections": [], "labels": [],
            "trace": {"cols": TRACE_COLS, "rows": []}}


# ---------------------------------------------------------------- routes

def clean_route_name(name):
    return " ".join(str(name or "").split())[:60]


class RouteBook:
    """walks/routes.json: every route you can navigate, its settings, and the current one.
    Each route is trained separately into walks/policies/<route>.json.

        {"current": "Halligan to Tisch",
         "routes": {"Halligan to Tisch": {"length_m": 120, "turn_min": 45, "created": "..."}}}
    """

    def __init__(self, root):
        self.root = root
        self.path = os.path.join(root, "routes.json")
        self.policy_dir = os.path.join(root, "policies")
        self.routes, self.current = {}, None
        try:
            with open(self.path) as f:
                d = json.load(f)
            self.routes, self.current = d.get("routes", {}), d.get("current")
        except (OSError, ValueError):
            pass

    def save(self):
        os.makedirs(self.root, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"current": self.current, "routes": self.routes}, f, indent=1)
        os.replace(tmp, self.path)

    def names(self):
        return sorted(self.routes, key=str.lower)

    def get(self, name):
        return self.routes.get(name)

    def create(self, name, length_m=None, turn_min=TURN_MIN_DEFAULT):
        name = clean_route_name(name)
        if not name:
            raise ValueError("give the route a name")
        if name in self.routes:
            raise ValueError(f"there is already a route called '{name}'")
        self.routes[name] = {"length_m": length_m, "turn_min": turn_min,
                             "created": time.strftime("%Y-%m-%d %H:%M")}
        if self.current is None:
            self.current = name
        self.save()
        return name

    def discover(self, names, turn_min=TURN_MIN_DEFAULT):
        """Register routes that only exist as walks (older recordings, copied-in walks)."""
        added = [n for n in names if n and n not in self.routes]
        for n in added:
            self.routes[n] = {"length_m": None, "turn_min": turn_min, "created": None}
        if self.current not in self.routes:
            self.current = self.names()[0] if self.routes else None
        if added:
            self.save()

    def update(self, name, **kw):
        r = self.routes[name]
        for key in ("length_m", "turn_min"):
            if key in kw:
                r[key] = kw[key]
        self.save()

    def rename(self, old, new):
        new = clean_route_name(new)
        if not new or new in self.routes:
            raise ValueError(f"'{new}' is empty or already taken")
        self.routes[new] = self.routes.pop(old)
        if os.path.exists(self.policy_path(old)):
            # The trained table inside is keyed by route name too: re-key it, keep the training.
            with open(self.policy_path(old)) as f:
                spec = json.load(f)
            spec["routes"] = {new if r == old else r: d for r, d in spec["routes"].items()}
            spec["route"] = new
            self.save_policy(new, spec)
            os.remove(self.policy_path(old))
        if self.current == old:
            self.current = new
        self.save()
        return new

    def delete(self, name):
        self.routes.pop(name, None)
        if os.path.exists(self.policy_path(name)):
            os.remove(self.policy_path(name))
        if self.current == name:
            self.current = self.names()[0] if self.routes else None
        self.save()

    def select(self, name):
        if name not in self.routes:
            raise ValueError(f"no route '{name}'")
        self.current = name
        self.save()

    def turn_min(self, name):
        return (self.routes.get(name) or {}).get("turn_min", TURN_MIN_DEFAULT)

    def lengths(self):
        return {n: r.get("length_m") for n, r in self.routes.items() if r.get("length_m")}

    def policy_path(self, name):
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "route"
        return os.path.join(self.policy_dir, f"{slug}-{hashlib.md5(name.encode()).hexdigest()[:6]}.json")

    def save_policy(self, name, spec):
        os.makedirs(self.policy_dir, exist_ok=True)
        tmp = self.policy_path(name) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(spec, f, separators=(",", ":"))
        os.replace(tmp, self.policy_path(name))

    def policy_info(self, name):
        try:
            with open(self.policy_path(name)) as f:
                spec = json.load(f)
        except (OSError, ValueError):
            return None
        return {"examples": sum(len(d["y"]) for d in spec["routes"].values()),
                "walks": spec.get("walks", []), "turn_min": spec.get("turn_min"),
                "calibration": spec.get("calibration", {}), "trained": spec.get("trained")}


# ---------------------------------------------------------------- policy

def dataset(walks, cal, turn_min=TURN_MIN_DEFAULT):
    """(route, x, action, weight) for every trace row of every walk. Rows inside an expert
    correction weigh 3x: they are exactly the states where the policy was wrong."""
    out = []
    for wi, w in enumerate(walks):
        rows, _ = derive(w, turn_min)
        times = [r["t"] for r in rows]
        labels = labels_per_row(times, w.get("labels", []))
        corr = labels_per_row(times, w.get("corrections", []))
        k = k_for(cal, w["walker"])
        for r, lab, c in zip(rows, labels, corr):
            if r["calibrating"]:
                continue
            expert = r.get("expert")
            weight = 3.0 if c != "straight" or (expert is not None and expert >= 0) else 1.0
            out.append((w["route"], features(r, k), lab, weight, wi))
    return out


class Policy:
    """Behaviour cloning by distance-weighted k-nearest neighbours over demonstration states,
    one table per route. With a handful of walks this beats a parametric model: it can't
    over-fit or under-fit, 'training' is instant on the board, and a DAgger correction
    takes effect immediately in the states around it."""

    def __init__(self, spec):
        self.k = spec.get("k", 7)
        self.scales = spec.get("scales", SCALES)
        self.turn_min = spec.get("turn_min", TURN_MIN_DEFAULT)
        self.calibration = spec.get("calibration", {})
        self.routes = {}
        self.ahead = {}      # route -> per row: the next two maneuvers [(action, metres ahead)]
        self._best = []
        for route, d in spec["routes"].items():
            self.routes[route] = [([v / s for v, s in zip(x, self.scales)], ACTIONS.index(a), wt)
                                  for x, a, wt in zip(d["X"], d["y"], d["w"])]
            self.ahead[route] = upcoming(d["X"], d["y"], d.get("walk"))

    @staticmethod
    def fit(walks, k=7, turn_min=TURN_MIN_DEFAULT, cal=None):
        cal = calibrate(walks) if cal is None else cal
        routes = {}
        for route, x, a, wt, wi in dataset(walks, cal, turn_min):
            d = routes.setdefault(route, {"X": [], "y": [], "w": [], "walk": []})
            d["X"].append([round(v, 3) for v in x])
            d["y"].append(a)
            d["w"].append(wt)
            d["walk"].append(wi)
        return {"type": "knn_policy", "k": k, "scales": SCALES, "features": FEATURES,
                "actions": ACTIONS, "turn_min": turn_min, "calibration": cal,
                "routes": routes, "walks": [w["id"] for w in walks]}

    def k_for(self, walker):
        return k_for(self.calibration, walker)

    def predict(self, route, x):
        """-> (action, confidence, votes)."""
        table = self.routes.get(route)
        if not table:
            return "straight", 0.0, {}
        q = [v / s for v, s in zip(x, self.scales)]
        best = []                                   # (d2, action, weight, row), k smallest
        worst = float("inf")
        for row, (xs, a, wt) in enumerate(table):
            d2 = 0.0
            for u, v in zip(xs, q):
                d2 += (u - v) * (u - v)
                if d2 >= worst:
                    break
            else:
                best.append((d2, a, wt, row))
                if len(best) > self.k:
                    best.sort()
                    best.pop()
                    worst = best[-1][0]
                elif len(best) == self.k:
                    worst = max(b[0] for b in best)
        self._best = (route, best)
        votes = [0.0] * len(ACTIONS)
        for d2, a, wt, _ in best:
            votes[a] += wt / (math.sqrt(d2) + 0.1)
        total = sum(votes) or 1.0
        i = max(range(len(votes)), key=votes.__getitem__)
        return ACTIONS[i], votes[i] / total, {ACTIONS[j]: round(v / total, 2)
                                             for j, v in enumerate(votes) if v > 0}


    def next_after(self, current):
        """What comes after `current` (the instruction now showing), from the neighbours of
        the last predict(): -> (action, metres ahead), or (None, None). Each neighbour knows
        the next two maneuvers on its own walk; if its next one is the instruction already
        showing (it hasn't reached it yet), the one after that counts."""
        route, best = self._best if self._best else (None, [])
        votes, dists = {}, {}
        for d2, _, wt, row in best:
            for a, ahead in self.ahead[route][row]:
                if a == current:
                    continue
                v = wt / (math.sqrt(d2) + 0.1)
                votes[a] = votes.get(a, 0.0) + v
                dists.setdefault(a, []).append((v, ahead))
                break
        if not votes:
            return None, None
        a = max(votes, key=votes.get)
        total = sum(v for v, _ in dists[a])
        return a, sum(v * d for v, d in dists[a]) / total


def upcoming(X, y, walk_of=None):
    """For every row of a policy table: the next two maneuvers (non-straight label
    stretches) after the stretch this row is in, as [(action, metres from this row to
    where it starts)]. `walk_of` says which walk each row came from; older policies
    lack it, so a new walk is assumed wherever the distance walked drops back."""
    n = len(y)
    if walk_of is None:
        walk_of, w = [], 0
        for i in range(n):
            if i and X[i][0] < X[i - 1][0] - 1.0:
                w += 1
            walk_of.append(w)
    out = [[] for _ in range(n)]
    i = 0
    while i < n:
        j = i
        while j < n and walk_of[j] == walk_of[i]:
            j += 1
        # rows i..j-1 are one walk: split into stretches of one label
        segs, s = [], i
        for r in range(i + 1, j + 1):
            if r == j or y[r] != y[s]:
                segs.append((s, r, y[s]))
                s = r
        for k, (s0, s1, _) in enumerate(segs):
            later = [(a, X[t][0]) for t, _, a in segs[k + 1:] if a != "straight"][:2]
            for r in range(s0, s1):
                out[r] = [(a, max(0.0, d - X[r][0])) for a, d in later]
        i = j
    return out


def load_policy(path):
    with open(path) as f:
        return Policy(json.load(f))


class Smoother:
    """Only switch the displayed instruction once the policy says the same thing twice in a
    row (0.4 s), so one odd neighbour doesn't flash a wrong arrow."""

    def __init__(self, hold=2):
        self.hold, self.shown, self.cand, self.n = hold, "straight", None, 0

    def push(self, action):
        if action == self.shown:
            self.cand, self.n = None, 0
        elif action == self.cand:
            self.n += 1
            if self.n >= self.hold:
                self.shown, self.cand, self.n = action, None, 0
        else:
            self.cand, self.n = action, 1
            if self.hold <= 1:
                self.shown = action
        return self.shown


# ---------------------------------------------------------------- evaluation

def evaluate(walks, k=7, turn_min=TURN_MIN_DEFAULT, eval_every=1, only_route=None, route_lengths=None):
    """Leave-one-walk-out: calibrate distance and train the policy without the held-out
    walk, replay its trace through the policy (with the live smoother), compare to its
    labels. Per walk: row accuracy, maneuvers announced (hit), how early, false alarms;
    and, if you typed its distance, the distance error of swing-sized steps vs a fixed
    step length (both calibrated on the other walks). `only_route`: hold out only that
    route's walks (the others still help calibrate each walker's stride)."""
    results = []
    for held in walks:
        if only_route is not None and held["route"] != only_route:
            continue
        others = [w for w in walks if w["id"] != held["id"]]
        cal = calibrate(others, route_lengths=route_lengths)
        res = {"id": held["id"], "route": held["route"], "walker": held["walker"]}
        if held.get("distance_m"):
            d = held["distance_m"]
            res["dist_err_swing"] = round(k_for(cal, held["walker"]) * _last(held, "swing") / d - 1, 3)
            res["dist_err_steps"] = round(
                k_for(calibrate(others, "steps", route_lengths), held["walker"], "steps")
                * _last(held, "steps") / d - 1, 3)
        train = [w for w in others if w["route"] == held["route"]]
        if not train:
            res["skipped"] = "only walk of this route"
            results.append(res)
            continue
        pol = Policy(Policy.fit(train, k=k, turn_min=turn_min, cal=cal))
        kk = pol.k_for(held["walker"])
        rows = [r for r in derive(held, turn_min)[0] if not r["calibrating"]]
        times = [r["t"] for r in rows]
        truth = labels_per_row(times, held.get("labels", []))
        sm, pred = Smoother(), []
        for i, r in enumerate(rows):
            if i % eval_every == 0:
                sm.push(pol.predict(held["route"], features(r, kk))[0])
            pred.append(sm.shown)
        correct = sum(p == t for p, t in zip(pred, truth))
        true_segs = segments_from_labels(times, truth)
        pred_segs = segments_from_labels(times, pred)
        hits, leads = 0, []
        for s in true_segs:
            idx = [i for i, t in enumerate(times) if s["t0"] <= t <= s["t1"] and pred[i] == s["action"]]
            # Announced early counts too, if it is still showing when the true segment begins.
            if idx:
                hits += 1
                j = idx[0]
                while j > 0 and pred[j - 1] == s["action"]:
                    j -= 1
                leads.append(round(s["t0"] - times[j], 1))
        false_alarms = sum(1 for p in pred_segs if not any(
            t["action"] == p["action"] and t["t0"] <= p["t1"] and p["t0"] <= t["t1"] for t in true_segs))
        res.update({"accuracy": round(correct / max(1, len(rows)), 3), "maneuvers": len(true_segs),
                    "hit": hits, "false_alarms": false_alarms, "lead_s": leads})
        results.append(res)
    return results
