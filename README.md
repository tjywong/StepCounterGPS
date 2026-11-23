# Gait Identifier + Campus Navigator

An ankle-worn Arduino UNO Q + Cytron MPU6050 that classifies **standing, walking, stairs up and stairs down** with two non-neural models (running and biking were dropped from the app; the step counter below was still trained on them, so it ignores pedalling):

| | Input | Classifier |
|---|---|---|
| **Model 1** | Peak acceleration along **one** signed axis in a 2.56 s window, measured from the window's mean so gravity drops out and the value reflects movement, not sensor tilt | Up to 3 thresholds (a 4-leaf decision tree on that single number). Training tries all six directions (±x, ±y, ±z) and keeps the best. |
| **Model 2** | All IMU outputs: accel x/y/z, gyro x/y/z, plus their magnitudes | Random forest on 86 features: mean, std, min, max, range, mean absolute deviation, zero-crossing rate, dominant cadence frequency and its power share, and jerk for each channel, plus axis correlations |

Both models run live on the board. The LED matrix shows **model 1 on the left, model 2 on the right**: `S` standing, `W` walking, `U` stairs up, `D` stairs down.

It also **counts steps with a neural network** trained in PyTorch (see [Step counter](#step-counter)). The count and cadence show on the control page.

## Hardware

| MPU6050 (Cytron) | UNO Q |
|---|---|
| VCC | 3.3V |
| GND | GND |
| SDA | SDA |
| SCL | SCL |

Leave AD0 unconnected (I2C address 0x68). Power the UNO Q from a USB-C power bank so it can be worn.

**Mounting matters.** Strap the IMU to the outside of the ankle, just above the ankle bone. Put it in the **same orientation every time**. The recordings in `data/` were made with the board's x axis pointing down the shin (`ax` ≈ −1 g when standing). The models learn from your recordings, so a rotated sensor looks like different data.

## How it works

```
MPU6050 --I2C 50 Hz--> MCU (sketch.ino) --Bridge, 10 samples/msg--> Linux (main.py)
                          ^                                            |
                          '-------- show_activity(m1, m2) --------------'
```

- `app/sketch/sketch.ino`: sets the MPU6050 to ±16 g / ±2000 °/s (fast steps at the ankle can exceed 8 g), turns on its 44 Hz low-pass filter, samples at 50 Hz, and streams raw readings. If reads fail for 1 s (a loose wire, or a sensor that lost power and restarted asleep), it clears the I2C bus, finds the sensor again at 0x68 or 0x69 and sets it up, and reports its health to the Linux side every 2 s (shown on both pages whenever the stream is under 40 Hz).
- `app/python/main.py`: serves the control page on port 7000 and starts in classify mode (`MODE` at the top). Record mode writes CSVs; classify mode slides a 128-sample window (64-sample hop), runs both models, and smooths each with a majority vote over 3 windows.
- `app/python/features.py`: feature extraction in pure Python, used **both** by the board and by training, so the numbers can't drift apart.
- `app/python/models.py`: pure-Python inference for the exported JSON models (no numpy or scikit-learn needed on the board).
- `app/python/stepnet.py`: pure-Python, one-sample-at-a-time inference for the step-counting network (`model_steps.json`).
- `training/train.py`: trains both activity models, cross-validates, and exports `app/python/model_peak.json` and `model_full.json`.
- `training/train_steps.py` + `training/step_labels.py`: label steps, train the step network in PyTorch, cross-validate, and export `app/python/model_steps.json`.

## Workflow

### 1. Record

1. Copy `app/` into App Lab as an app. On the board, apps live in `~/ArduinoApps/`.
2. Run the app and open the control page at `http://<board-ip>:7000` on your phone or laptop. Pick the **activity** and press **Record**.
3. Do that one activity, then press **Idle** (or pick the next activity, which starts a new file). Each recording is saved as `recordings/<label>_<time>.csv` in the app folder. The page shows how many seconds you have, a count per activity, and a delete button for bad takes. `MODE` / `RECORD_LABEL` at the top of `main.py` only set what happens at startup.
4. Record **at least 2 sessions (3+ is better) of 1–3 minutes per activity**, on different occasions. **Stairs:** the shipped models were trained on `data_tylerunoq/` (standing, walking and about 40 s each of stairs up and down, recorded on TylerUnoQ) with `python training/train.py --data data_tylerunoq --trim 2`. The older `data/` was recorded with a different mounting, and mixing the two made model 2 call a quarter of the walking "stairs down". Training takes a recording's activity from its folder name, so keep stairs files in `stairs_up/` and `stairs_down/` folders. For more stairs data, record `Stairs ↑` and `Stairs ↓` sessions on a few different staircases (walk up a flight, stop the recording or switch to the other direction at the top), count your steps, and retrain both `train.py` and `train_steps.py`. Stairs count steps like walking. Short flights of 3–4 steps are shorter than the classifier's 2.56 s window and will mostly be missed. For biking, include air-biking sessions if you'll test with air-biking. The first and last 5 s of each file are discarded.
5. Optional, for the step counter: while walking or running, count your steps, then type the total into the **steps** box next to that file on the page. Counts are saved in `recordings/step_counts.csv`.

### 2. Train (on your laptop)

```bash
scp -r arduino@<board>.local:ArduinoApps/<app>/recordings/* data/   # or adb pull
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python training/train.py
```

Files are labelled by folder (`data/walking/…`) or by file-name prefix (`walking_….csv`). The script prints and saves to `training/report.txt`:
- the cross-validated score for each direction model 1 could use, plus the chosen thresholds in g
- confusion matrices and precision/recall for both models
- model 2's most important features
- a check that the exported pure-Python models agree with scikit-learn

Cross-validation is **grouped by recording**: windows from one session never appear in both the training and the test set. Overlapping windows from the same session would otherwise make the score look better than it is. If an activity has only one recording, the script falls back to a within-session split and warns you.

Then train the step counter. It trains 3 networks per held-out recording plus the final 3, about 15 minutes with an Apple-silicon GPU (it uses `mps`/CUDA when available) and much longer on CPU alone:

```bash
.venv/bin/python training/train_steps.py            # add --no-cv to just train and export
```

It writes `app/python/model_steps.json` and `training/report_steps.txt`.

### 3. Deploy

Copy the new `model_peak.json`, `model_full.json` and `model_steps.json` into the app's `python/` folder, run the app again, and press **Classify** on the control page. The page and the LED matrix show both models' predictions live, and the page shows the step count, cadence and a **Reset** button. The console prints model 1's peak value, both predictions and the step count every 1.28 s. Without `model_steps.json` the app still classifies but doesn't count steps.

## Step counter

**Input: every motion sensor on the board.** The UNO Q has no IMU of its own, so that means all six MPU6050 axes (accel x/y/z, gyro x/y/z) plus the accel and gyro magnitudes, 8 channels at 50 Hz. The MPU6050's temperature sensor is left out on purpose: it changes slowly, says nothing about steps, and would let the network tell recording sessions apart instead of learning gait.

**Network:** three causal temporal convolutional networks (4.9k parameters each) trained from different random starts, with their outputs averaged. Each has a pointwise input layer, six residual blocks of dilated convolutions (kernel 3, dilations 1–32, 2.5 s of history), and a ReLU output that gives a *step density* per sample. The running sum is the step count, so there's no peak picking or threshold. Each step is reported 0.5 s after it happens, so the network sees the whole stride before deciding.

Averaging matters because, with this little training data, identical networks trained from different starts count noticeably differently (held-out running ranged from 75 to 113 of 176 steps across runs). The average is steadier, and one bad network costs a third of the count instead of all of it. During training the output leaks a little below zero (leaky ReLU), so a network can't get stuck outputting zero forever. That failure happened before this fix.

The networks run one sample at a time in pure Python on the board: about 1 ms per sample for all three on an M2 laptop, so expect a noticeable share of one of the UNO Q's cores at 50 Hz. Train with `--members 1` for a lighter single network. `train_steps.py` checks that the export matches PyTorch (they agree to about 3e-6) and refuses to write it if they differ by more than 1e-3.

**Labels** (`training/step_labels.py`), in order of preference:
1. a per-sample `step` column (synthetic data has one);
2. an offline stride detector, rescaled to your hand count if you entered one. At the ankle, each stride of the instrumented leg is one burst of shank rotation. The detector finds the main gyro axis with PCA, low-pass filters it, and takes one peak per stride. It sees the whole recording and knows the activity, which the board can't do live. Each stride is 2 steps, placed at the instrumented leg's mid-swing (the event the sensor actually sees). On the synthetic data the detector is within 1–2 steps of the true marks per recording.

Standing and biking are always 0 steps. A plain pedometer counts every pedal stroke as a stride, and learning not to is most of what the network is for.

**Training:** 20 s sequences stitched from random crops of random activities, so the network also sees transitions. Augmentation: up to ±60° of mounting rotation, gyro bias, gain changes, noise, and crops played 0.55–1.8× as fast. The speed range lets walking stand in for running cadences. **Sensor dropout:** 30% of training sequences hide all accelerometer or all gyro channels, so each network learns both the foot-strike impacts (accel) and the leg swing (gyro) instead of leaning on one. The loss combines per-sample density, 2 s window counts and whole-sequence counts.

**Accuracy** (`training/report_steps.txt`): each recording is counted by networks that never saw it, compared with a plain gyro-threshold pedometer (0.2 s average of |gyro|, one stride per rise through 120 °/s). The first 2.5 s (warm-up) and last 0.5 s of each recording are left out for every method.

| held-out recordings | step counter (shipped) | plain pedometer |
|---|---|---|
| walking, 3 sessions (244 steps) | 252 counted, **3.4%** total error (+0%, +10%, +0%) | 4.1% |
| running, 1 session (170 steps) | 136 counted, −20% | −38% |
| standing + biking, 5 sessions (3.3 min) | **0.5 false steps/min** | 57 false steps/min |

What the network clearly adds is ignoring pedalling and fidgeting, plus a better running count. **Running is the weak spot:** there's only one running recording, so when it's held out the networks have never seen running. Record 2–3 more running sessions (with hand counts) and retrain; that's the single biggest improvement available. The "true" counts come from the stride detector unless you've entered hand counts, so hand counts are also the only fully independent check.

**Sensors compared** (3 averaged networks each, same held-out test):

| held out | accel + gyro, sensor dropout (shipped) | accel only (`--inputs accel`) | accel + gyro, no dropout |
|---|---|---|---|
| walking, total error | **3.4%** | 3.8% | 7.8%* |
| running (170 steps) | −20% | **+7%** | −37%* |
| standing + biking, false steps/min | **0.5** | 1.5 | 0.7* |

\* scored before the warm-up fix, so walking is a few steps low.

Without sensor dropout, the gyro dominates and the count falls apart on running, which this data never shows the networks. Accelerometer impacts carry over to running much better. Dropout makes the all-sensor model use both. It's the best on walking and on ignoring biking, but with one running recording, accel-only still counts running better. Once you have a few running recordings, retrain both (`--inputs accel` vs the default) and keep whichever wins.

## Campus navigator

A behaviour-cloned guide that tells you **straight / turn left / turn right / stairs up / stairs down / arrived** along a route you've demonstrated, using only the ankle IMU: no GPS, wifi, camera or magnetometer. Open `http://<board-ip>:7000/nav.html` on your phone (there's a **Navigator ›** link on the gait page).

### Routes

You can keep as many routes as you like. The **route picker** at the top of every tab shows them (✓ trained, ↻ needs retraining, number of walks), and **＋ New** creates one with a name and, optionally, its length in metres. The board remembers the current route, so both partners' phones always show the same one. Everything you do applies to the picked route:
- **Walk** records demos and navigates it.
- **Label** lists only its walks.
- **Train** trains it alone into its own policy, and shows its settings and held-out check.

Each route has its own **length** (calibrates stride for everyone who walks it) and **turn threshold**. Settings are on the Train tab, along with **rename** (walks and training follow) and **delete** (removes its walks too). You can switch routes at any time; switching during a walk asks first, then stops and saves that walk.

**The 10-walk limit applies per route:** each route's navigator is trained on at most 10 walks (demos + DAgger runs, both of you). If your assignment means 10 walks in total across all routes, set `MAX_TRAIN_WALKS` in `app/python/nav.py` and keep to one route. Stride calibration is per walker and pools walks from every route, since your stride doesn't depend on where you walk.

### How it works

```
IMU 50 Hz ─┬─> step counter (neural net) ───────────> steps ─┐
           ├─> gyro across gravity: shank swing ─────────────┴─> distance = Σ step × swing × k
           ├─> gyro about gravity, bias calibrated ─> heading ─> turn detector (threshold)
           └─> gait random forest ─> walking / stairs up / stairs down / ...
every 0.2 s: observation = (distance, distance since last turn, heading, turns so far)
             policy = 7 nearest demonstration moments of this route ─> instruction
```

- **Heading** (`app/python/nav.py`) integrates the gyro component along gravity (low-passed accelerometer), so it works however the strap is rotated, and positive is always a left turn. Each walk starts with **2 s standing still** to measure gyro bias (≈6 °/s on this MPU6050, which would otherwise be 360° of drift per minute). Whenever the foot is planted, heading stops integrating and the bias is re-learned. On the recordings in `data/`, the heading barely changes when the gravity low-pass constant goes from 0.5 s to ∞, so shank swing doesn't leak into it.
- **Distance from shank swing, not a fixed step length.** A longer step swings the shank through a bigger angle (pendulum: step ≈ leg length × swing angle), so slowing down in a crowd shortens your steps on the board too. The gyro component *across* gravity is the swing rate. Over the last ~2 s, swing ÷ steps counted gives degrees per step (≈60° on the walking recordings in `data/`). Every step the neural counter reports is sized by that, delayed 0.5 s to line up with the counter. Standing and biking add nothing, because the step counter gives them zero steps. One number per walker, **k** (metres per degree of swing), turns it into metres.
- **Calibrating k: tell it how far you walked.** Give the route a **length** (when creating it, or under route settings), which then applies to every walk of it, or type a single walk's **Distance walked** on the Label tab. Measure it on a campus map, or walk a stretch you've measured. Per walker, k comes from the first that applies:
  1. **measured:** that walker's walks with a typed distance;
  2. **route length:** walks on a route someone measured;
  3. **matched to partner:** a route your partner walked, assumed the same length;
  4. **default:** a 0.70 m step at 60° of swing.

  The Train tab shows each walker's k as "step at 60° swing" and where it came from. Live distances show `≈` until you're calibrated.
- **Turn detector:** a heading change of at least the route's **turn threshold** (default 45°, set per route on the Train tab) that settles. Smaller changes are wiggles: sidesteps, swerving round people, and the few degrees of foot wobble each stride. Gentle path curves are followed and stay "straight". Raw heading is also a policy input. The detector exists because heading drifts: turns so far and distance since the last turn give checkpoints that don't, and the labels need turn start and end times. Turn counts are recomputed from each walk's stored heading, so a new threshold applies to old walks too. After changing it, redo the auto-labels and train; the policy remembers the threshold it was trained with and navigates with that one.
- **Stairs classifier:** the gait random forest (model 2) runs every 1.28 s during walks, and its output is stored with each walk. Once it's trained on stairs (see Workflow above), its stairs stretches become auto-labels (thin strip under the label band on the timeline) and show live as "on stairs". Stairs you tap during a walk take priority where both exist. The classifier reports stairs about 2.5 s late; the labels are shifted back to make up for it.
- **Policy:** distance-weighted k-nearest-neighbours (k = 7) over every 0.2 s of the training walks of that route. With ≤10 walks a parametric model has too little to fit, while kNN can't over- or under-fit, trains instantly on the board, and a DAgger correction takes effect in exactly the states around it. Corrected moments count 3×. Live instructions change only after two agreeing predictions (0.4 s).
- **What's next:** every training moment also knows the next two maneuvers on its own walk and how far ahead they were. The same 7 neighbours vote on what comes after the instruction now showing and average its distance, so the screen shows e.g. "Then ⇣ Stairs down in 8 m" (smoothed, counting down as you walk), and the phone says "Now turn left. In 8 metres, go down the stairs."

### Labels: generated afterwards, fixed before training

Labelling during a walk is hard, so you don't. After a demo the board **labels it in hindsight**: you actually walked the route, so the turns the gyro saw *are* the expert's actions. Each turn and each flight of stairs is labelled from 6 m before it starts until it ends ("Announce turns __ m early" on the Label tab). Stairs come from your taps and from the classifier, and the last 2 m are "arrived". Corrections from DAgger runs are applied on top. The **Label** tab then shows the walk as a dead-reckoned **map** and a **timeline** (heading, detected turns ▲, label band, classifier stairs, and for DAgger runs what the policy said and what you corrected). Drag on the timeline or tap the map to select a stretch and paint it with the right action. You can also redo the auto-labels with a different lead distance or after typing the walk's distance. **Save & use for training** ticks the walk. At most **10 walks per route** (demos + DAgger runs, both of you together) can be ticked; the board enforces it.

### Workflow

1. **＋ New** route (name, and its length if you know it), or pick one. **Walk tab:** enter your name. Stand still, press **● Record demo**, wait for "measuring gyro bias" to clear (if it says "No sensor data" or "Sensor dropping out" instead, fix the wiring first), and walk the route. Tap **Up starts / Down starts / Stairs end** at stairs (until the classifier knows stairs, the taps are the only stairs labels). Press **■ Stop & save** at the destination.
2. **Label tab** opens on that walk. Check the map looks like the route, enter the distance walked if you didn't before, fix any wrong labels (a missed turn, a sidestep detected as a turn), then **Save & use for training**.
3. Both partners demo the route (e.g. 2 demos each = 4 walks), then **Train** tab → **Train '<route>'**. Repeat 1–3 for each route; each trains separately. The leave-one-walk-out table shows, per walk, how often the instruction matched, how many maneuvers were announced, and false alarms. For walks with a typed distance it also shows the **distance error** of swing-sized steps vs a fixed step length, each calibrated only on the other walks: the check of whether swing is helping.
4. **DAgger:** **▶ Navigate** the route and follow the instructions honestly. When one is wrong, tap what you should do; the screen shows your correction instead (a left/right correction ends once you've made that turn, others when you tap **Done correcting**). With **Speak** on, the phone says each new instruction and vibrates, so you can keep it in your pocket. The run is saved like a demo: review it, save it, and train again. Use your remaining walks this way on the routes or moments the policy gets wrong.

Walks live in `walks/` in the app folder (`<id>.json` = metadata, labels and 5 Hz trace of steps, swing, heading and activity; `<id>.imu.csv` = raw IMU). Routes, their settings and the current route are in `walks/routes.json`; each route's policy is `walks/policies/<route>.json`. To check on the laptop (every route, or `--route "<name>"`):

```bash
scp -r arduino@<board>.local:ArduinoApps/<app>/walks .
python3 training/nav_eval.py --walks walks      # writes training/report_nav.txt
```

### Synthetic test (pipeline only)

`training/nav_sim.py` builds walks from the **real** ankle recordings in `data/`: walking and standing stretches are stitched into two campus-like routes (3 turns plus a flight of stairs each). Those recordings circle the room, so their rotation about gravity is projected out first; then known turns and a random gyro bias are injected. Two walkers have different step lengths and cadences. Only walker A's first walk of each route has its distance typed in, so walker B is calibrated from the route length. Everything downstream is the board's code.

```bash
python3 training/nav_sim.py && python3 training/nav_eval.py --walks data_nav_synthetic
```

With 8 walks (2 routes × 2 walkers × 2), every turn is detected and estimated route lengths are within ~10% of the truth. Held-out walks get **37 of 40 maneuvers announced, 1 false alarm**, at turn thresholds of 30°, 45° or 70° alike (the injected turns are ~90°). Most announcements fall inside the 4 m lead-in, before the turn begins. Swing-sized and fixed-length steps score the same here (5.1% vs 5.5% distance error), because the simulator only slows strides down; it can't make them longer or shorter. **Only your walks at different paces can show whether swing helps**: check the distance errors on the Train tab. All of this tests the software, not real-world accuracy.

### Limitations

- **Route must be chosen;** the device doesn't know where you are. Start every walk at the same spot, facing the same way.
- **Distance drift:** expect a few % error even when calibrated. Turns act as checkpoints and *distance since the last turn* keeps errors local, but long legs with no turn are where early or late instructions come from. **Arrival** is the last 2 m of the demos, so a walker whose distance comes up a few metres short may not be told "arrived"; that's a good DAgger correction to make.
- **Swing vs. stride length is unverified on real data:** the recordings in `data/` have no measured distances (and "running" there, done circling a room, shows no more swing per step than walking).
- **Stairs need recordings** before the classifier knows them; until then, tap them.
- **Detours:** an extra turn (around a crowd) shifts *turns so far*. Heading and distance usually still pick the right neighbours, but a correction here is good DAgger data.

## Synthetic data (pipeline test only)

`training/simulate.py` generates hand-made approximations of ankle signals, so the code can be tested end to end without hardware:

```bash
.venv/bin/python training/simulate.py && .venv/bin/python training/train.py --data data_synthetic
```

The `model_*.json` files in `app/python/` are trained on the real recordings in `data/`, not on this synthetic data. Retrain on your own recordings if you mount the sensor differently.

Synthetic recordings also carry a `step` column with the true steps, so `train_steps.py --data data_synthetic` checks the step counter and the stride detector against exact answers.

## Expected limitations

- **Model 1, standing vs. biking:** a single peak value hardly separates slow pedalling from standing. Expect most of model 1's errors here and between walking and biking. Model 2 uses gyro rotation and cadence, so it should separate them much better.
- **Real biking vs. air-biking:** these put gravity on different axes. If you'll use both, record both under `biking`.
- **Transitions:** windows spanning a change of activity are ambiguous. The 3-window vote adds about 2.5 s of lag.
- **Step count:** steps appear 0.5 s late, and the first ~2.5 s after starting Classify are a warm-up. Starting or stopping a walk can be off by one step, because the detector can't tell which foot moves first.
