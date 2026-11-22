"""Step labels for training the step-counting network (laptop only).

Ground truth for each recording, in order of preference:
  1. a per-sample `step` column (1 on the sample a step happens) -- synthetic data has this;
  2. otherwise the offline stride detector below, rescaled to the hand-counted total in
     data/step_counts.csv when you've entered one on the control page.
Standing and biking recordings always get zero steps, whatever the detector says.
Stairs (up and down) count steps like walking.

The detector works on the whole recording at once (non-causal) and knows the activity,
which is exactly what the board can't do live: that's the network's job.

At the ankle, each stride of the instrumented leg is one burst of shank rotation (swing)
between two still stance phases. Each burst is one stride = two steps: the instrumented
foot's step at mid-swing, and the other foot's step halfway to the next mid-swing.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 50
STEPPING = {"walking", "running", "stairs_up", "stairs_down"}
COUNTS_FILE = "step_counts.csv"


def gyro_swing(rows):
    """Low-passed shank rotation rate (deg/s) about the recording's main gyro axis, signed so
    the forward swing is positive: one peak per stride, whatever the mounting. (The gyro
    magnitude would peak twice per stride if the backward swing were as strong.)"""
    g = np.asarray(rows)[:, 3:6]
    g = g - g.mean(axis=0)
    axis = np.linalg.svd(g, full_matrices=False)[2][0]
    s = g @ axis
    if np.percentile(s, 99) < -np.percentile(s, 1):
        s = -s
    b, a = butter(2, 3.0 / (FS / 2))
    return filtfilt(b, a, s)


def stride_period(swing):
    """Typical stride period (samples) from the autocorrelation, searched over 0.5-2 s."""
    x = swing - swing.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo, hi = int(0.5 * FS), int(2.0 * FS)
    if len(ac) <= hi:
        return FS
    return lo + int(np.argmax(ac[lo:hi]))


def detect_strides(rows):
    """Sample indices of mid-swing of the instrumented leg."""
    swing = gyro_swing(rows)
    period = stride_period(swing)
    top = np.percentile(swing, 98)
    peaks, _ = find_peaks(swing, distance=max(1, int(0.6 * period)),
                          height=max(50.0, 0.35 * top), prominence=0.3 * top)
    return peaks, period


def strides_to_steps(strides, period):
    """Two steps per stride. Within a bout, the other foot steps halfway between mid-swings;
    the bout's last stride gets its partner half a period later."""
    steps = []
    for i, s in enumerate(strides):
        steps.append(s)
        nxt = strides[i + 1] if i + 1 < len(strides) else None
        gap = (nxt - s) if nxt is not None else None
        steps.append(s + (gap // 2 if gap is not None and gap < 1.7 * period else period // 2))
    return np.array(sorted(steps), dtype=int)


def manual_counts(data_dir):
    """{file name: hand-counted steps} from step_counts.csv anywhere under data_dir."""
    out = {}
    for path in Path(data_dir).rglob(COUNTS_FILE):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("steps", "").strip():
                    out[r["file"].strip()] = float(r["steps"])
    return out


def step_events(rows, label, marks=None, per_stride=False):
    """Returns (event array, per-event weight, source). Weights let a hand-counted
    total rescale the detector's events without moving them.

    per_stride=True puts both of a stride's steps on the instrumented leg's mid-swing
    (weight 2), an event the sensor actually sees, instead of guessing when the other
    foot lands. Totals are the same."""
    if marks is not None:
        idx = np.flatnonzero(np.asarray(marks) > 0)
        return idx, np.ones(len(idx)), "marked"
    if label not in STEPPING:
        return np.zeros(0, dtype=int), np.zeros(0), "none"
    strides, period = detect_strides(rows)
    if per_stride:
        return strides, np.full(len(strides), 2.0), "detector"
    steps = strides_to_steps(strides, period)
    steps = steps[steps < len(rows)]
    return steps, np.ones(len(steps)), "detector"
