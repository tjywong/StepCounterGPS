"""UNO Q Linux side: receives IMU samples from the MCU over the Bridge and
either records them to CSV (for training), classifies the activity and counts
steps live, or runs the campus navigator (demonstrations, navigation with DAgger
corrections, label review and training).

Open http://<board-ip>:7000 for the gait page and http://<board-ip>:7000/nav.html for
the navigator. MODE / RECORD_LABEL below only set what happens at startup.
"""
import collections
import json
import os
import threading
import time
from pathlib import Path

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

import features
import models
import nav
import stepnet

# ---------------------------------------------------------------- settings
MODE = "classify"        # at startup: "idle", "record" or "classify"
RECORD_LABEL = "walking"  # at startup; used in the file name while recording
SMOOTH = 3               # majority vote over this many consecutive windows

ACCEL_LSB_PER_G = 2048.0    # MPU6050 at +-16 g (set in the sketch)
GYRO_LSB_PER_DPS = 16.4     # MPU6050 at +-2000 deg/s

HERE = Path(__file__).resolve().parent
RECORD_DIR = HERE.parent / "recordings"
COUNTS_FILE = RECORD_DIR / "step_counts.csv"   # hand-counted steps per recording, for training
MODES = ("idle", "record", "classify")
WALK_DIR = HERE.parent / "walks"                # navigator: demonstrations, DAgger runs, policy
# Routes, their settings and the current route: walks/routes.json; policies: walks/policies/.
STEP_MODEL = HERE / "model_steps.json"

# Codes understood by show_activity() in the sketch; -1 blanks that half.
CODE = {name: i for i, name in enumerate(features.ACTIVITIES)}

# ---------------------------------------------------------------- state
lock = threading.Lock()
pending = collections.deque()   # (t_ms, ax, ay, az, gx, gy, gz) in g / dps
received = 0


def on_imu_batch(batch):
    """Bridge handler. batch = 't,ax,ay,az,gx,gy,gz;' repeated, raw ints."""
    global received
    rows = []
    for rec in batch.split(";"):
        if not rec:
            continue
        p = rec.split(",")
        if len(p) != 7:
            continue
        t, ax, ay, az, gx, gy, gz = (int(v) for v in p)
        rows.append((t, ax / ACCEL_LSB_PER_G, ay / ACCEL_LSB_PER_G, az / ACCEL_LSB_PER_G,
                     gx / GYRO_LSB_PER_DPS, gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS))
    with lock:
        pending.extend(rows)
        received += len(rows)


Bridge.provide("imu_batch", on_imu_batch)

# The sketch reports sensor health every 2 s while reads fail, and once when they recover.
imu_health = ""


def on_imu_health(text):
    global imu_health
    if text != imu_health:
        print("IMU:", text)
    imu_health = text


Bridge.provide("imu_health", on_imu_health)


def drain():
    with lock:
        rows = list(pending)
        pending.clear()
    return rows


def show(peak_code, full_code):
    try:
        Bridge.call("show_activity", peak_code, full_code)
    except Exception as e:  # display is optional; never stop the app over it
        print("show_activity failed:", e)


# ---------------------------------------------------------------- record mode
class Recorder:
    def __init__(self, label):
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.label = label
        self.path = RECORD_DIR / f"{label}_{stamp}.csv"
        self.f = open(self.path, "w")
        self.f.write("t_ms,ax,ay,az,gx,gy,gz\n")
        self.samples = 0
        print(f"Recording '{label}' to {self.path}")

    def step(self):
        rows = drain()
        for r in rows:
            self.f.write("%d,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f\n" % r)
        self.f.flush()
        self.samples += len(rows)
        time.sleep(0.2)

    def close(self):
        self.f.close()
        print(f"Saved {self.path.name}: {self.samples / features.FS:.0f} s")

    def status(self):
        return {"file": self.path.name, "seconds": round(self.samples / features.FS, 1)}


# ---------------------------------------------------------------- classify mode
class Classifier:
    def __init__(self):
        self.peak = models.load(HERE / "model_peak.json")
        self.full = models.load(HERE / "model_full.json")
        self.buf = collections.deque(maxlen=features.WINDOW)
        self.since_last = 0
        self.hist_peak = collections.deque(maxlen=SMOOTH)
        self.hist_full = collections.deque(maxlen=SMOOTH)
        self.last = None
        print(f"Model 1 uses peak accel along {self.peak.direction}; "
              f"model 2 is a {len(self.full.trees)}-tree random forest")
        try:
            self.steps = stepnet.load(HERE / "model_steps.json")
            print(f"Step counter: neural net, steps appear {self.steps.delay / features.FS:.1f} s late")
        except FileNotFoundError:
            self.steps = None
            print("No model_steps.json: not counting steps (run training/train_steps.py)")

    @staticmethod
    def _vote(hist):
        return collections.Counter(hist).most_common(1)[0][0]

    def step(self):
        for r in drain():
            if self.steps:
                self.steps.push(r[1:])
            self.buf.append(r[1:])
            self.since_last += 1
            if len(self.buf) == features.WINDOW and self.since_last >= features.HOP:
                self.since_last = 0
                self._classify(list(self.buf))
        time.sleep(0.05)

    def _classify(self, window):
        peak = self.peak.feature(window)
        self.hist_peak.append(self.peak.predict(window))
        self.hist_full.append(self.full.predict(window))
        s1, s2 = self._vote(self.hist_peak), self._vote(self.hist_full)
        self.last = {"peak_g": round(peak, 2), "model1": s1, "model2": s2}
        steps = f"  steps={self.steps.total:.0f} ({self.steps.cadence:.0f}/min)" if self.steps else ""
        print(f"peak={peak:5.2f} g  model1(peak)={s1:<9} model2(all)={s2:<9}{steps}")
        show(CODE[s1], CODE[s2])

    def close(self):
        show(-1, -1)

    def reset_steps(self):
        if self.steps:
            self.steps.reset()

    def status(self):
        steps = ({"steps": int(self.steps.total), "cadence": round(self.steps.cadence)}
                 if self.steps else {"steps": None})
        return {"direction": self.peak.direction, "prediction": self.last,
                "warming_up": self.last is None, **steps}


# ---------------------------------------------------------------- navigator
class NavSession:
    """One walk of a route. kind 'demo': you walk it, the board records odometry (labels
    are generated afterwards). kind 'dagger': the policy gives instructions and you
    follow them; tap a correction whenever it's wrong. Both are saved as walks.

    Distance = shank-swing-sized steps x k, k = this walker's calibration (metres per
    degree of swing), fitted from walks where a distance was typed in."""

    def __init__(self, store, kind, route, walker, distance_m, turn_min, cal, policy_path):
        if kind not in ("demo", "dagger"):
            raise ValueError(kind)
        if not route:
            raise ValueError("create or pick a route first")
        self.store, self.kind = store, kind
        self.policy = None
        if kind == "dagger":
            if not os.path.exists(policy_path):
                raise ValueError(f"'{route}' isn't trained yet (Train tab)")
            self.policy = nav.load_policy(policy_path)
            if route not in self.policy.routes:
                raise ValueError(f"the policy has no demonstrations of '{route}'")
            # Features must be computed exactly as in training: its threshold and calibration.
            turn_min, cal = self.policy.turn_min, self.policy.calibration
        self.walk = nav.new_walk(kind, route, str(walker)[:30] or "me", distance_m)
        self.turn_min = turn_min
        self.k = nav.k_for(cal, self.walk["walker"])
        self.calibrated = cal.get(self.walk["walker"], {}).get("source", "default") != "default"
        self.odo = nav.Odometry(stepnet.load(STEP_MODEL), turn_min=turn_min,
                                classify=self._load_classifier())
        self.smoother = nav.Smoother()
        # What comes after the current instruction: shown action, and where along the walk
        # (metres) it starts, smoothed so the countdown doesn't jump around.
        self.next_smoother = nav.Smoother(hold=3)
        self.next_action, self.next_at = None, None
        self.row = None
        self.action, self.policy_action, self.conf, self.votes = "straight", None, 0.0, {}
        self.override = None            # expert correction in force: {"action", "t0"}
        self.changed = False
        self.saved = None
        self.raw = open(store.root + "/" + self.walk["id"] + ".imu.csv", "w")
        self.raw.write("t_ms,ax,ay,az,gx,gy,gz\n")
        print(f"Navigator {kind}: '{route}' walker {self.walk['walker']} "
              f"k={self.k * 60:.2f} m per 60 deg swing, turns >= {turn_min:.0f} deg")

    @staticmethod
    def _load_classifier():
        """The gait random forest (model 2): it reports stairs once it's trained on them."""
        try:
            model = models.load(HERE / "model_full.json")
        except (OSError, ValueError) as e:
            print("Navigator: no activity classifier:", e)
            return None
        if not set(nav.STAIRS) & set(model.classes):
            print("Navigator: model_full.json has no stairs classes yet (record stairs and retrain)")
        return model.predict

    def step(self):
        for r in drain():
            self.raw.write("%d,%.5f,%.5f,%.5f,%.3f,%.3f,%.3f\n" % r)
            row = self.odo.push(r[1:])
            if row:
                self._on_row(row)
        time.sleep(0.05)

    def _on_row(self, row):
        prev = self.row
        self.row = row
        before = self.action
        pol = exp = -1
        if self.policy and not row["calibrating"]:
            a, self.conf, self.votes = self.policy.predict(self.walk["route"], nav.features(row, self.k))
            self.policy_action = self.smoother.push(a)
            pol = nav.ACTIONS.index(self.policy_action)
            self.action = self.policy_action
        if self.override and prev and row["n_turns"] > prev["n_turns"]:
            # A left/right correction ends by itself once that turn has been made.
            o = self.override["action"]
            if o in ("left", "right") and (self.odo.events[-1]["angle"] > 0) == (o == "left"):
                self.end_correction(row["t"])
        if self.override:
            self.action = self.override["action"]
            exp = nav.ACTIONS.index(self.action)
        if self.policy and not row["calibrating"]:
            self._update_next(row)
        self.walk["trace"]["rows"].append([row["t"], row["steps"], row["swing"], row["heading"],
                                           row["activity"], row["calibrating"], pol, exp])
        self.changed = self.changed or self.action != before

    def _update_next(self, row):
        here = row["swing"] * self.k
        a, ahead = self.policy.next_after(self.action)
        shown = self.next_smoother.push(a or "none")
        if shown == "none" or shown == self.action:
            self.next_action, self.next_at = None, None
            return
        if shown != self.next_action:
            self.next_action, self.next_at = shown, None
        if a == shown:
            at = here + ahead
            self.next_at = at if self.next_at is None else self.next_at + 0.3 * (at - self.next_at)

    def correct(self, action):
        """Expert takes over with `action`; None hands control back to the policy."""
        t = self.row["t"] if self.row else 0.0
        if self.override:
            self.end_correction(t)
        if action in nav.ACTIONS:
            self.override = {"action": action, "t0": t}
        self.changed = True

    def end_correction(self, t):
        o = self.override
        self.walk["corrections"].append({"t0": o["t0"], "t1": t, "action": o["action"]})
        self.override = None

    def mark(self, kind):
        if kind in ("stairs_up", "stairs_down", "stairs_end") and self.row:
            self.walk["marks"].append({"t": self.row["t"], "kind": kind})

    def close(self):
        self.raw.close()
        if self.override and self.row:
            self.end_correction(self.row["t"])
        if len(self.walk["trace"]["rows"]) < 5 * 5:   # < 5 s: a false start, don't keep it
            self.store.delete(self.walk["id"])
            print("Navigator: walk too short, discarded")
            return
        self.walk["labels"], _ = nav.auto_label(self.walk, self.k, self.turn_min)
        self.store.save(self.walk)     # not ticked for training until you review it
        self.saved = self.walk["id"]
        print(f"Navigator: saved {self.walk['id']}")

    def status(self):
        r = self.row or {}
        return {"nav": {
            "kind": self.kind, "id": self.walk["id"], "route": self.walk["route"],
            "walker": self.walk["walker"], "calibrated": self.calibrated,
            "calibrating": bool(r.get("calibrating", 1)), "t": r.get("t", 0),
            "steps": int(r.get("steps", 0)), "dist": round(r.get("swing", 0) * self.k, 1),
            "step_m": round(r.get("swing_per_step", nav.DEFAULT_SWING_PER_STEP) * self.k, 2),
            "heading": round(r.get("heading", 0)), "turns": r.get("n_turns", 0),
            "activity": r.get("activity", ""),
            "action": self.action, "policy_action": self.policy_action,
            "override": self.override["action"] if self.override else None,
            "confidence": round(self.conf, 2), "votes": self.votes,
            "next": self.next_action,
            "next_m": None if self.next_at is None
            else max(0, round(self.next_at - r.get("swing", 0) * self.k)),
            "marks": self.walk["marks"][-3:], "corrections": len(self.walk["corrections"]),
        }}

class Idle:
    def step(self):
        drain()
        time.sleep(0.2)

    def close(self):
        pass

    def status(self):
        return {}


# ---------------------------------------------------------------- web control page
ui = WebUI()


class Controller:
    """Owns the current mode. Web handlers only queue requests; the app loop
    applies them, so the recorder/classifier is only touched from one thread."""

    def __init__(self, mode, label):
        self.mode, self.label = "idle", label
        self.worker = Idle()
        self.error = None
        self.requests = collections.deque()
        self.recordings = []
        self.counts = self._load_counts()
        self.rate_hz, self._rate_t, self._rate_n = 0.0, time.time(), 0
        self._last_push = 0.0
        self.store = nav.WalkStore(str(WALK_DIR))
        self.walks, self.last_saved, self.nav_msg = [], None, None
        self.eval, self.evaluating = {}, False     # eval: {route: leave-one-walk-out results}
        self.cal = {}
        self.routes = nav.RouteBook(str(WALK_DIR))
        self._scan_walks()
        self._switch(mode, label)

    def request(self, **kwargs):
        with lock:
            self.requests.append(kwargs)

    def _switch(self, mode, label, nav_args=None):
        old = self.worker
        old.close()
        if isinstance(old, NavSession):
            self.last_saved = old.saved
            self._scan_walks()
        self.worker, self.mode, self.label = Idle(), "idle", label
        drain()  # don't let samples from before the switch leak into the new file
        try:
            if mode == "record":
                self.worker = Recorder(label)
            elif mode == "classify":
                self.worker = Classifier()
            elif mode in ("demo", "dagger"):
                self.worker = NavSession(self.store, mode, **nav_args)
            self.mode, self.error, self.nav_msg = mode, None, None
        except Exception as e:  # e.g. model files missing
            self.error = f"Couldn't start {mode}: {e}"
            if mode in ("demo", "dagger"):
                self.nav_msg = self.error
            print(self.error)
        self._scan_recordings()

    # ------------------------------------------------------------ navigator
    def _scan_walks(self):
        """Walk list, routes and stride calibration. Calibration (shown, and used for labels
        and live distance) uses every walk with a known distance on any route; training a
        route uses ticked walks only."""
        walks = self.store.all()
        self.routes.discover({w["route"] for w in walks}, self._legacy_turn_min())
        self.cal = nav.calibrate(walks, route_lengths=self.routes.lengths())
        self.walks = sorted((nav.WalkStore.summary(w, nav.k_for(self.cal, w["walker"])) for w in walks),
                            key=lambda w: w["id"], reverse=True)

    @staticmethod
    def _legacy_turn_min():
        """Turn threshold from the single settings.json of the previous version, if any."""
        try:
            with open(WALK_DIR / "settings.json") as f:
                return float(json.load(f)["turn_min"])
        except (OSError, ValueError, KeyError, TypeError):
            return nav.TURN_MIN_DEFAULT

    def _route_status(self):
        out = []
        for name in self.routes.names():
            r = self.routes.get(name)
            mine = [w for w in self.walks if w["route"] == name]
            pol = self.routes.policy_info(name)
            ticked = sorted(w["id"] for w in mine if w["include"])
            out.append({"name": name, "length_m": r.get("length_m"), "turn_min": r.get("turn_min"),
                        "walks": len(mine), "ticked": len(ticked),
                        "trained": pol["trained"] if pol else None,
                        "examples": pol["examples"] if pol else 0,
                        "policy_turn_min": pol["turn_min"] if pol else None,
                        # stale: the ticked walks or the threshold changed since training
                        "stale": bool(pol) and (sorted(pol["walks"]) != ticked
                                                or pol["turn_min"] != r.get("turn_min"))})
        return out

    def _send_walk(self, walk, proposed=False):
        tm = self.routes.turn_min(walk["route"])
        _, turns = nav.derive(walk, tm)
        ui.send_message("walk", {"walk": walk, "turns": turns, "proposed": proposed,
                                 "k": nav.k_for(self.cal, walk["walker"]),
                                 "cal": self.cal.get(walk["walker"]),
                                 "turn_min": tm, "actions": nav.ACTIONS})

    def _n_included(self, route, except_id=None):
        return sum(1 for w in self.walks if w["include"] and w["route"] == route and w["id"] != except_id)

    def _set_include(self, walk, include):
        if include and self._n_included(walk["route"], walk["id"]) >= nav.MAX_TRAIN_WALKS:
            self.nav_msg = (f"'{walk['route']}' already has {nav.MAX_TRAIN_WALKS} training walks: "
                            "untick one first.")
            return False
        walk["include"] = bool(include)
        return True

    @staticmethod
    def _metres(v):
        """A typed distance: positive number of metres, or None for 'don't know'."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return round(v, 1) if 1 <= v <= 20000 else None

    def _apply_route(self, req):
        """Route book requests: create, select, update (length, threshold, rename), delete."""
        try:
            if "route_create" in req:
                a = req["route_create"] or {}
                name = self.routes.create(a.get("name"), self._metres(a.get("length_m")),
                                          self._legacy_turn_min())
                self._select(name)
                self.nav_msg = f"Created '{name}'."
            elif "route_select" in req:
                self._select(req["route_select"])
            elif "route_update" in req:
                a = req["route_update"] or {}
                name = a.get("name")
                if name not in self.routes.routes:
                    raise ValueError(f"no route '{name}'")
                if "length_m" in a:
                    self.routes.update(name, length_m=self._metres(a["length_m"]))
                if a.get("turn_min") not in (None, ""):
                    tm = min(120.0, max(15.0, float(a["turn_min"])))
                    if tm != self.routes.turn_min(name):
                        self.routes.update(name, turn_min=tm)
                        self.nav_msg = (f"'{name}' now counts turns from {tm:.0f}°. Redo the auto-labels "
                                        "on its walks, then train it again.")
                clean = nav.clean_route_name(a.get("new_name", ""))
                if clean and clean != name:
                    if isinstance(self.worker, NavSession) and self.worker.walk["route"] == name:
                        raise ValueError("stop the walk on this route before renaming it")
                    new = self.routes.rename(name, clean)
                    for w in self.store.all():
                        if w["route"] == name:
                            w["route"] = new
                            self.store.save(w)
                    if name in self.eval:
                        self.eval[new] = self.eval.pop(name)
                    self.nav_msg = f"Renamed to '{new}'."
            elif "route_delete" in req:
                name = req["route_delete"]
                if isinstance(self.worker, NavSession) and self.worker.walk["route"] == name:
                    raise ValueError("stop the walk on this route before deleting it")
                n = 0
                for w in self.store.all():
                    if w["route"] == name:
                        self.store.delete(w["id"])
                        n += 1
                self.routes.delete(name)
                self.eval.pop(name, None)
                self.nav_msg = f"Deleted '{name}' and its {n} walks."
            else:
                return False
        except (ValueError, TypeError, KeyError) as e:
            self.nav_msg = str(e)
        self._scan_walks()
        return True

    def _select(self, name):
        """Pick the current route. A walk in progress on another route is stopped and saved."""
        if name not in self.routes.routes:
            raise ValueError(f"no route '{name}'")
        if isinstance(self.worker, NavSession) and self.worker.walk["route"] != name:
            self._switch("idle", self.label)
            if self.last_saved:
                self.nav_msg = f"Saved the walk on the previous route; now on '{name}'."
        self.routes.select(name)

    def _apply_nav(self, req):
        """Navigator requests. Returns True if handled."""
        if any(key.startswith("route_") for key in req):
            return self._apply_route(req)
        if "nav_start" in req:
            a = req["nav_start"]
            route = self.routes.current
            if a.get("route") and a["route"] != route:   # page is behind: another phone switched route
                self.nav_msg = f"The route is now '{route}' (changed on another phone). Check it and start again."
                return True
            r = self.routes.get(route) or {}
            args = {"route": route, "walker": a.get("walker", ""),
                    "distance_m": r.get("length_m"),
                    "turn_min": self.routes.turn_min(route), "cal": self.cal,
                    "policy_path": self.routes.policy_path(route) if route else ""}
            self._switch(a.get("kind"), self.label, args)
            return True
        if "nav_stop" in req:
            if isinstance(self.worker, NavSession):
                self._switch("idle", self.label)
                if self.last_saved:
                    self._send_walk(self.store.load(self.last_saved))
            return True
        if "nav_mark" in req:
            if isinstance(self.worker, NavSession):
                self.worker.mark(req["nav_mark"])
            return True
        if "nav_correct" in req:
            if isinstance(self.worker, NavSession):
                self.worker.correct(req["nav_correct"])
            return True
        if "train" in req:
            self._train(req["train"] if isinstance(req["train"], str) else self.routes.current)
            return True
        wid = req.get("walk_get") or req.get("walk_autolabel") or req.get("walk_save") \
            or req.get("walk_include") or req.get("walk_delete")
        if wid is None:
            return False
        if wid not in self.store.ids():
            self.nav_msg = f"no walk {wid}"
            return True
        if isinstance(self.worker, NavSession) and self.worker.walk["id"] == wid:
            self.nav_msg = "that walk is still being recorded"
            return True
        if "walk_delete" in req:
            self.store.delete(wid)
            self._scan_walks()
            return True
        walk = self.store.load(wid)
        if "walk_get" in req:
            self._send_walk(walk)
        elif "walk_autolabel" in req:
            try:
                lead = min(15.0, max(0.0, float(req.get("lead_m", nav.LEAD_M_DEFAULT))))
            except (TypeError, ValueError):
                lead = nav.LEAD_M_DEFAULT
            cal = self.cal
            if "distance_m" in req:      # use the distance being typed in, even before saving
                walk["distance_m"] = self._metres(req.get("distance_m"))
                cal = nav.calibrate([w for w in self.store.all() if w["id"] != wid] + [walk],
                                    route_lengths=self.routes.lengths())
            walk["labels"], _ = nav.auto_label(walk, nav.k_for(cal, walk["walker"]),
                                               self.routes.turn_min(walk["route"]), lead_m=lead)
            self._send_walk(walk, proposed=True)
        elif "walk_include" in req:
            if self._set_include(walk, req.get("include")):
                self.store.save(walk)
            self._scan_walks()
        elif "walk_save" in req:
            labels = []
            for seg in req.get("labels") or []:
                try:
                    if seg["action"] in nav.ACTIONS:
                        labels.append({"t0": float(seg["t0"]), "t1": float(seg["t1"]),
                                       "action": seg["action"]})
                except (KeyError, TypeError, ValueError):
                    pass
            walk["labels"], walk["reviewed"] = labels, True
            walk["distance_m"] = self._metres(req.get("distance_m"))
            self._set_include(walk, req.get("include", True))
            self.store.save(walk)
            self._scan_walks()
            self._send_walk(walk)
        return True

    def _train(self, route):
        """Train one route on its ticked walks. Stride calibration uses ticked walks of every
        route: a walker's stride doesn't depend on where they walk."""
        if route not in self.routes.routes:
            self.nav_msg = "create or pick a route first"
            return
        ticked = [w for w in self.store.all() if w.get("include")]
        walks = [w for w in ticked if w["route"] == route]
        if not walks:
            self.nav_msg = f"no walks of '{route}' ticked for training yet"
            return
        tm = self.routes.turn_min(route)
        cal = nav.calibrate(ticked, route_lengths=self.routes.lengths())
        spec = nav.Policy.fit(walks, turn_min=tm, cal=cal)
        spec["trained"] = time.strftime("%Y-%m-%d %H:%M")
        spec["route"] = route
        self.routes.save_policy(route, spec)
        self.nav_msg = f"Trained '{route}' on {len(walks)} walks (turns ≥ {tm:.0f}°)."
        print(self.nav_msg)
        # Leave-one-walk-out check in the background: a few seconds of pure Python.
        if not self.evaluating:
            self.evaluating, self.eval[route] = route, None
            threading.Thread(target=self._evaluate, args=(ticked, route, tm, self.routes.lengths()),
                             daemon=True).start()

    def _evaluate(self, walks, route, turn_min, lengths):
        try:
            self.eval[route] = nav.evaluate(walks, turn_min=turn_min, eval_every=2,
                                            only_route=route, route_lengths=lengths)
        except Exception as e:
            self.eval[route] = [{"id": "-", "route": route, "skipped": f"evaluation failed: {e}"}]
        self.evaluating = False

    @staticmethod
    def _load_counts():
        counts = {}
        if COUNTS_FILE.exists():
            with open(COUNTS_FILE) as f:
                for line in f.read().splitlines()[1:]:
                    name, _, steps = line.partition(",")
                    if steps.strip():
                        counts[name] = int(steps)
        return counts

    def _save_counts(self):
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        with open(COUNTS_FILE, "w") as f:
            f.write("file,steps\n")
            for name, steps in sorted(self.counts.items()):
                f.write(f"{name},{steps}\n")

    def _apply(self, req):
        if self._apply_nav(req):
            return
        if "reset_steps" in req:
            if isinstance(self.worker, Classifier):
                self.worker.reset_steps()
            return
        if "set_steps" in req:
            name, steps = req["set_steps"], req.get("steps")
            if name not in {r["name"] for r in self.recordings}:
                return
            if steps is None or str(steps).strip() == "":
                self.counts.pop(name, None)
            else:
                try:
                    self.counts[name] = max(0, int(float(steps)))
                except ValueError:
                    return
            self._save_counts()
            self._scan_recordings()
            return
        if "delete" in req:
            names = {p.name for p in RECORD_DIR.glob("*.csv")}
            current = self.worker.path.name if isinstance(self.worker, Recorder) else None
            if req["delete"] in names and req["delete"] != current:
                (RECORD_DIR / req["delete"]).unlink()
                print(f"Deleted {req['delete']}")
                if self.counts.pop(req["delete"], None) is not None:
                    self._save_counts()
                self._scan_recordings()
            return
        mode = req.get("mode", self.mode)
        label = req.get("label", self.label)
        if mode not in MODES + ("demo", "dagger") or label not in features.ACTIVITIES:
            return
        if mode in ("demo", "dagger"):      # navigator sessions start only via nav_start
            if mode != self.mode:
                return
            self.label = label
            return
        # A new label while recording starts a new file, so each file has one activity.
        if mode != self.mode or (mode == "record" and label != self.label):
            self._switch(mode, label)
        else:
            self.label = label

    def _scan_recordings(self):
        out = []
        for p in sorted(RECORD_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
            if p == COUNTS_FILE:
                continue
            with open(p) as f:
                n = sum(1 for _ in f) - 1
            out.append({"name": p.name, "seconds": round(n / features.FS),
                        "steps": self.counts.get(p.name)})
        self.recordings = out

    def status(self):
        return {"mode": self.mode, "label": self.label, "error": self.error,
                "rate_hz": self.rate_hz, "imu_health": imu_health, "labels": features.ACTIVITIES,
                "recordings": self.recordings,
                "walks": self.walks, "max_walks": nav.MAX_TRAIN_WALKS,
                "routes": self._route_status(), "route": self.routes.current,
                "eval": self.eval, "evaluating": self.evaluating,
                "last_saved": self.last_saved, "nav_msg": self.nav_msg, "calibration": self.cal,
                **self.worker.status()}

    def step(self):
        with lock:
            reqs = list(self.requests)
            self.requests.clear()
        for r in reqs:
            self._apply(r)
        self.worker.step()

        now = time.time()
        if now - self._rate_t >= 2:
            self.rate_hz = round((received - self._rate_n) / (now - self._rate_t), 1)
            self._rate_t, self._rate_n = now, received
        changed = isinstance(self.worker, NavSession) and self.worker.changed
        if changed or now - self._last_push >= 0.5:
            if changed:
                self.worker.changed = False
            self._last_push = now
            ui.send_message("status", self.status())


ctl = Controller(MODE, RECORD_LABEL)

ui.on_connect(lambda sid: ctl.request())  # empty request -> fresh status soon
ui.on_message("set_mode", lambda sid, data: ctl.request(mode=data.get("mode")))
ui.on_message("set_label", lambda sid, data: ctl.request(label=data.get("label")))
ui.on_message("delete_recording", lambda sid, data: ctl.request(delete=data.get("name")))
ui.on_message("set_steps", lambda sid, data: ctl.request(set_steps=data.get("name"), steps=data.get("steps")))
ui.on_message("reset_steps", lambda sid, data: ctl.request(reset_steps=True))
# navigator
ui.on_message("nav_start", lambda sid, data: ctl.request(nav_start=data or {}))
ui.on_message("nav_stop", lambda sid, data: ctl.request(nav_stop=True))
ui.on_message("nav_mark", lambda sid, data: ctl.request(nav_mark=(data or {}).get("kind")))
ui.on_message("nav_correct", lambda sid, data: ctl.request(nav_correct=(data or {}).get("action") or ""))
ui.on_message("walk_get", lambda sid, data: ctl.request(walk_get=(data or {}).get("id")))
ui.on_message("walk_autolabel", lambda sid, data: ctl.request(
    walk_autolabel=(data or {}).get("id"), lead_m=(data or {}).get("lead_m"),
    distance_m=(data or {}).get("distance_m")))
ui.on_message("walk_save", lambda sid, data: ctl.request(
    walk_save=(data or {}).get("id"), labels=(data or {}).get("labels"),
    include=(data or {}).get("include", True), distance_m=(data or {}).get("distance_m")))
ui.on_message("walk_include", lambda sid, data: ctl.request(
    walk_include=(data or {}).get("id"), include=(data or {}).get("include")))
ui.on_message("walk_delete", lambda sid, data: ctl.request(walk_delete=(data or {}).get("id")))
ui.on_message("train", lambda sid, data: ctl.request(train=(data or {}).get("route") or True))
ui.on_message("route_create", lambda sid, data: ctl.request(route_create=data or {}))
ui.on_message("route_select", lambda sid, data: ctl.request(route_select=(data or {}).get("name")))
ui.on_message("route_update", lambda sid, data: ctl.request(route_update=data or {}))
ui.on_message("route_delete", lambda sid, data: ctl.request(route_delete=(data or {}).get("name")))


def loop():
    ctl.step()


App.run(user_loop=loop)
