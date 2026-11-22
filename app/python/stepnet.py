"""Step counter: a small causal convolutional network, run one sample at a time.

Trained in PyTorch by training/train_steps.py, exported to model_steps.json, and
evaluated here in pure Python (no numpy or torch on the board).

Input per sample: all six IMU axes plus the accel and gyro magnitudes. Output: a step
density (steps per sample). Summing it gives the step count, so there's no peak picking
or threshold to tune. The network looks at the last ~2.5 s and is trained to report each
step `delay` samples after it happens, so it can see how the stride finishes.

Each network: pointwise input layer -> residual blocks of dilated causal convolutions
(kernel 3, dilations 1, 2, 4, ...) with ReLU -> pointwise output -> ReLU, so standing
still can give exactly zero. The model file holds several networks; their outputs are averaged.
"""
import collections
import json
import math
from operator import mul


def inputs(sample):
    """(ax, ay, az, gx, gy, gz) in g and deg/s -> the network's 8 input channels."""
    ax, ay, az, gx, gy, gz = sample
    return [ax, ay, az, gx, gy, gz,
            math.sqrt(ax * ax + ay * ay + az * az), math.sqrt(gx * gx + gy * gy + gz * gz)]


class Network:
    """One trained network, run one sample at a time."""

    def __init__(self, spec):
        self.mean, self.std = spec["mean"], spec["std"]
        self.in_w, self.in_b = spec["in_w"], spec["in_b"]          # [hidden][8], [hidden]
        self.blocks = []
        for blk in spec["blocks"]:
            k, d = blk["kernel"], blk["dilation"]
            # Flatten each output's weights to match the history read oldest tap first:
            # torch weight[o][c][j] multiplies x[t - (k-1-j)*d][c].
            flat = [[blk["w"][o][c][j] for j in range(k) for c in range(len(blk["w"][o]))]
                    for o in range(len(blk["w"]))]
            hidden = len(self.in_b)
            hist = collections.deque([[0.0] * hidden for _ in range((k - 1) * d + 1)],
                                     maxlen=(k - 1) * d + 1)
            self.blocks.append((k, d, flat, blk["b"], hist))
        self.out_w, self.out_b = spec["out_w"], spec["out_b"]

    def push(self, channels):
        x = [(v - m) / s for v, m, s in zip(channels, self.mean, self.std)]
        h = [b + sum(map(mul, w, x)) for w, b in zip(self.in_w, self.in_b)]
        for k, d, flat, bias, hist in self.blocks:
            hist.append(h)
            taps = []
            for j in range(k):
                taps.extend(hist[j * d])
            h = [v + max(0.0, b + sum(map(mul, w, taps))) for v, w, b in zip(h, flat, bias)]
        return max(0.0, self.out_b + sum(map(mul, self.out_w, h)))


class StepCounter:
    """Averages the step density of several networks trained from different random starts.
    With this little training data, one network can go silent on an unfamiliar recording;
    the average then loses a fraction of the count instead of all of it."""

    def __init__(self, spec):
        if spec.get("type") == "step_tcn":          # a single network
            members = [spec]
        elif spec.get("type") == "step_tcn_ensemble":
            members = spec["members"]
        else:
            raise ValueError("model_steps.json is not a step-counter model")
        self.fs = spec["fs"]
        self.delay = spec["delay"]
        self.receptive_field = spec["receptive_field"]
        self.nets = [Network(m) for m in members]
        self.total = 0.0
        self.seen = 0
        self.recent = collections.deque(maxlen=10 * self.fs)   # densities, for cadence

    def push(self, sample):
        """Feed one sample; returns the step density for the step `delay` samples ago.
        Nothing counts until the networks' history is full (~2.5 s): before that they see
        the zeros they start with, not real motion."""
        ch = inputs(sample)
        density = sum(net.push(ch) for net in self.nets) / len(self.nets)
        self.seen += 1
        if self.seen <= self.receptive_field:
            density = 0.0
        self.total += density
        self.recent.append(density)
        return density

    @property
    def cadence(self):
        """Steps per minute over the last 10 s."""
        if not self.recent:
            return 0.0
        return sum(self.recent) * 60.0 * self.fs / len(self.recent)

    def reset(self):
        self.total = 0.0


def load(path):
    with open(path) as f:
        return StepCounter(json.load(f))
