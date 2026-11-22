# Gait Identifier

An ankle-worn Arduino UNO Q + Cytron MPU6050 that classifies **standing, walking, running and biking** (real biking, or lying on your back and air-biking) with two non-neural models:

| | Input | Classifier |
|---|---|---|
| **Model 1** | Peak acceleration along **one** signed axis in a 2.56 s window, measured from the window's mean so gravity drops out and the value reflects movement, not sensor tilt | Up to 3 thresholds (a 4-leaf decision tree on that single number). Training tries all six directions (±x, ±y, ±z) and keeps the best. |
| **Model 2** | All IMU outputs: accel x/y/z, gyro x/y/z, plus their magnitudes | Random forest on 86 features: mean, std, min, max, range, mean absolute deviation, zero-crossing rate, dominant cadence frequency and its power share, and jerk for each channel, plus axis correlations |

Both models run live on the board. The LED matrix shows **model 1 on the left, model 2 on the right**: `S` standing, `W` walking, `R` running, `B` biking.

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

- `app/sketch/sketch.ino`: sets the MPU6050 to ±16 g / ±2000 °/s (running impacts at the ankle exceed 8 g), turns on its 44 Hz low-pass filter, samples at 50 Hz, and streams raw readings.
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
4. Record **at least 2 sessions (3+ is better) of 1–3 minutes per activity**, on different occasions. For biking, include air-biking sessions if you'll test with air-biking. The first and last 5 s of each file are discarded.
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
