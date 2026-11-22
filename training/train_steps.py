"""Train the step-counting network in PyTorch and export it for the board.

    python training/train_steps.py                 # reads data/, writes app/python/model_steps.json
    python training/train_steps.py --data data_synthetic
    python training/train_steps.py --no-cv         # skip cross-validation, just train and export

Labels come from training/step_labels.py (marked steps, hand counts, or the offline stride
detector; standing and biking are always zero). Accuracy is measured leave-one-recording-
out: each recording is counted by networks that never saw it, streamed from the start
exactly as the board would. The report compares it with a plain gyro-threshold pedometer
so you can see what the network adds.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app" / "python"))
sys.path.insert(0, str(ROOT / "training"))
import features      # noqa: E402
import step_labels   # noqa: E402
import stepnet       # noqa: E402

FS = features.FS
DELAY = 25           # report each step 0.5 s late, so the network sees the whole stride
SIGMA = 2.0          # samples; each step is a small Gaussian bump that sums to 1
SEQ = 1000           # training sequence length (20 s)
WARP_MAX = 1.8       # crops are played up to this much faster or slower (--warp)
ROT_MAX = 60.0       # degrees of random mounting rotation per sequence (--rot)
PER_STRIDE = True    # 2 steps at each mid-swing (--target strides) rather than one per foot
HIDDEN, KERNEL, DILATIONS = 16, 3, (1, 2, 4, 8, 16, 32)
RF = 1 + (KERNEL - 1) * sum(DILATIONS)
N_IN = 8
MEMBERS = 3          # networks averaged in the exported model (--members)
SENSOR_DROPOUT = 0.3  # chance a training sequence hides all accel or all gyro (--sensor-dropout)
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
INPUT_SETS = {"all": range(8), "accel": (0, 1, 2, 6), "gyro": (3, 4, 5, 7)}   # for --inputs


# ---------------------------------------------------------------- data
def label_for(path):
    if path.parent.name in features.ACTIVITIES:
        return path.parent.name
    # Longest matching prefix: 'stairs_up_2026...' is stairs_up, not 'stairs'.
    hits = [c for c in features.ACTIVITIES if path.stem.startswith(c + "_")]
    return max(hits, key=len) if hits else None


def load_recordings(data_dir, trim_s):
    counts = step_labels.manual_counts(data_dir)
    trim = int(trim_s * FS)
    recs = []
    for path in sorted(Path(data_dir).rglob("*.csv")):
        label = label_for(path)
        if label is None or path.name == step_labels.COUNTS_FILE:
            continue
        with open(path) as f:
            reader = list(csv.DictReader(f))
        rows = np.array([[float(r[k]) for k in ("ax", "ay", "az", "gx", "gy", "gz")] for r in reader])
        marks = [int(r["step"]) for r in reader] if reader and "step" in reader[0] else None
        events, weights, source = step_labels.step_events(rows, label, marks, PER_STRIDE)
        manual = counts.get(path.name)
        if manual is not None and source != "marked":
            if len(events):
                weights = weights * manual / len(events)
                source = f"detector scaled to hand count {manual:g}"
            elif manual > 0:
                print(f"  {path.name}: hand count {manual:g} ignored, no strides found to place it on")
        keep = (events >= trim) & (events < len(rows) - trim)
        events, weights = events[keep] - trim, weights[keep]
        rows = rows[trim:len(rows) - trim]
        if len(rows) < 2 * RF:
            continue
        density = np.zeros(len(rows))
        t = np.arange(len(rows))
        for e, w in zip(events, weights):
            lo, hi = max(0, e - 8), min(len(rows), e + 9)
            bump = np.exp(-0.5 * ((t[lo:hi] - e) / SIGMA) ** 2)
            density[lo:hi] += w * bump / bump.sum()
        recs.append({"name": path.name, "label": label, "rows": rows, "density": density,
                     "events": events, "weights": weights, "source": source})
        print(f"  {label:<9} {path.name}: {len(rows) / FS:5.0f} s, "
              f"{weights.sum():5.0f} steps ({source})")
    return recs


def channels(rows):
    """(N, 6) -> (N, 8), same channels as stepnet.inputs."""
    acc, gyr = rows[:, :3], rows[:, 3:6]
    return np.concatenate([rows, np.linalg.norm(acc, axis=1, keepdims=True),
                           np.linalg.norm(gyr, axis=1, keepdims=True)], axis=1)


# ---------------------------------------------------------------- augmentation
def rotation(rng, max_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    a = math.radians(rng.uniform(-max_deg, max_deg))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def warp(rows, density, f):
    """Play the crop f times faster: resample time, scale rotation rates, keep step totals."""
    n = len(rows)
    m = max(2, int(n / f))
    pos = np.minimum(np.arange(m) * f, n - 1)
    out = np.stack([np.interp(pos, np.arange(n), rows[:, i]) for i in range(6)], axis=1)
    out[:, 3:6] *= f
    cum = np.concatenate([[0.0], np.cumsum(density)])
    edges = np.minimum(np.arange(m + 1) * f, n)
    return out, np.diff(np.interp(edges, np.arange(n + 1), cum))


def make_sequence(recs, by_label, rng):
    """SEQ samples stitched from random crops of random activities, so the network also
    sees transitions. One mounting rotation and gyro bias per sequence, like a real session."""
    parts_x, parts_d, n = [], [], 0
    while n < SEQ:
        label = rng.choice(list(by_label))
        r = recs[rng.choice(by_label[label])]
        length = int(rng.uniform(3, 20) * FS)
        start = rng.integers(0, max(1, len(r["rows"]) - length))
        x, d = r["rows"][start:start + length], r["density"][start:start + length]
        x, d = warp(x, d, math.exp(rng.uniform(-math.log(WARP_MAX), math.log(WARP_MAX))))
        parts_x.append(x)
        parts_d.append(d)
        n += len(x)
    x, d = np.concatenate(parts_x)[:SEQ], np.concatenate(parts_d)[:SEQ]
    rot = rotation(rng, ROT_MAX)
    x = np.concatenate([x[:, :3] @ rot.T * rng.uniform(0.95, 1.05),
                        x[:, 3:6] @ rot.T * rng.uniform(0.9, 1.1) + rng.normal(0, 3, size=3)], axis=1)
    x[:, :3] += rng.normal(0, 0.01, size=(SEQ, 3))
    x[:, 3:6] += rng.normal(0, 0.5, size=(SEQ, 3))
    target = np.concatenate([np.zeros(DELAY), d[:SEQ - DELAY]])
    return channels(x), target


# ---------------------------------------------------------------- model
class StepTCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = nn.Conv1d(N_IN, HIDDEN, 1)
        self.convs = nn.ModuleList(nn.Conv1d(HIDDEN, HIDDEN, KERNEL, dilation=d) for d in DILATIONS)
        self.out = nn.Conv1d(HIDDEN, 1, 1)
        self.register_buffer("mean", torch.zeros(N_IN))
        self.register_buffer("std", torch.ones(N_IN))
        self.register_buffer("mask", torch.ones(N_IN))   # 0 = channel hidden from the network

    def forward(self, x, drop=None):
        """x: (batch, time, 8) raw channels -> (batch, time) step density.
        drop: optional (batch, 8) 0/1 per sequence, for sensor dropout in training."""
        x = (x - self.mean) / self.std * self.mask
        if drop is not None:
            x = x * drop[:, None, :]
        x = x.transpose(1, 2)
        h = self.inp(x)
        for conv in self.convs:
            h = h + F.relu(conv(F.pad(h, ((KERNEL - 1) * conv.dilation[0], 0))))
        z = self.out(h).squeeze(1)
        # A plain ReLU output that goes negative for every input passes no gradient, so the
        # network can die and count zero forever. Training leaks a little below zero to keep
        # it alive; inference (and the board) clamps at zero.
        return F.leaky_relu(z, 0.01) if self.training else F.relu(z)


def loss_fn(pred, target):
    pred, target = pred[:, RF:], target[:, RF:]     # skip the zero-padded warm-up
    dense = F.mse_loss(pred, target) * 100
    win = 2 * FS
    pw = F.avg_pool1d(pred[:, None], win, stride=FS // 2) * win
    tw = F.avg_pool1d(target[:, None], win, stride=FS // 2) * win
    whole = (pred.sum(1) - target.sum(1)).abs().mean() / 10
    return dense + (pw - tw).abs().mean() + whole


def train(recs, iters, seed, log=False, inputs="all"):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    by_label = {}
    for i, r in enumerate(recs):
        by_label.setdefault(r["label"], []).append(i)
    model = StepTCN()
    model.mask.zero_()
    model.mask[list(INPUT_SETS[inputs])] = 1.0
    allx = channels(np.concatenate([r["rows"] for r in recs]))
    model.mean.copy_(torch.tensor(allx.mean(0), dtype=torch.float32))
    model.std.copy_(torch.tensor(allx.std(0) + 1e-3, dtype=torch.float32))
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=iters)
    for it in range(iters):
        batch = [make_sequence(recs, by_label, rng) for _ in range(32)]
        x = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32, device=DEVICE)
        y = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32, device=DEVICE)
        drop = torch.ones(len(batch), N_IN)
        for b in range(len(batch)):
            if rng.random() < SENSOR_DROPOUT:
                drop[b, list(INPUT_SETS["gyro" if rng.random() < 0.5 else "accel"])] = 0.0
        loss = loss_fn(model(x, drop.to(DEVICE)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log and (it + 1) % 200 == 0:
            print(f"    iter {it + 1}/{iters}  loss {loss.item():.4f}")
    return model.cpu().eval()


def train_ensemble(recs, iters, seed, members, log=False, inputs="all"):
    """Several networks from different random starts; their densities get averaged."""
    models = []
    for m in range(members):
        if log:
            print(f"  network {m + 1}/{members}")
        models.append(train(recs, iters, seed=seed * 1000 + m, log=log, inputs=inputs))
    return models


@torch.no_grad()
def predict(models, rows):
    """Average step density of the networks over a whole recording, as if streamed from power-on."""
    x = torch.tensor(channels(rows)[None], dtype=torch.float32)
    d = np.mean([m(x)[0].numpy() for m in models], axis=0)
    d[:RF] = 0.0   # warm-up, as on the board (stepnet.StepCounter)
    return d


# ---------------------------------------------------------------- baseline
def baseline_density(rows):
    """Classic ankle pedometer: 0.2 s moving average of |gyro|; each rise through 120 deg/s
    (re-armed below 60, at most one per 0.4 s) is a stride = 2 steps. Causal, no delay."""
    g = np.linalg.norm(rows[:, 3:6], axis=1)
    k = FS // 5
    s = np.convolve(g, np.ones(k) / k)[:len(g)]
    out, armed, last = np.zeros(len(g)), True, -FS
    for i, v in enumerate(s):
        if armed and v > 120 and i - last >= 0.4 * FS:
            out[i], armed, last = 2.0, False, i
        elif v < 60:
            armed = True
    return out


# ---------------------------------------------------------------- evaluation
def score(rec, density, delay):
    """True and counted steps over the recording, plus mean |error| per 10 s block.
    Steps in the last `delay` samples haven't been reported yet, and nothing counts during
    the network's warm-up (the first RF samples), so both are left out, for every method."""
    n = len(rec["rows"])
    truth = np.zeros(n)
    np.add.at(truth, rec["events"], rec["weights"])
    truth = np.concatenate([np.zeros(delay), truth[:n - delay]])
    truth[:RF] = 0.0
    density = density.copy()
    density[:RF] = 0.0
    block = 10 * FS
    nb = (n - RF) // block
    errs = [abs(density[RF + i * block:RF + (i + 1) * block].sum()
                - truth[RF + i * block:RF + (i + 1) * block].sum()) for i in range(nb)]
    return truth.sum(), density.sum(), float(np.mean(errs)) if errs else 0.0


def summarize(title, results):
    lines = [f"== {title} ==",
             f"{'recording':<34}{'activity':<10}{'true':>7}{'net':>8}{'err%':>7}{'|err|/10s':>10}"
             f"{'baseline':>10}{'err%':>7}"]
    for name, label, t, p, e10, b, _ in results:
        pct = lambda c: f"{100 * (c - t) / t:+6.1f}" if t >= 1 else "     -"
        lines.append(f"{name:<34}{label:<10}{t:7.0f}{p:8.1f}{pct(p):>7}{e10:10.2f}{b:10.0f}{pct(b):>7}")
    stepping = [r for r in results if r[1] in step_labels.STEPPING]
    still = [r for r in results if r[1] not in step_labels.STEPPING]
    if stepping:
        tt = sum(r[2] for r in stepping)
        lines.append(f"walking/running: net counts {sum(r[3] for r in stepping):.0f} of {tt:.0f} steps, "
                     f"total |error| {100 * sum(abs(r[3] - r[2]) for r in stepping) / tt:.1f}% "
                     f"(baseline {100 * sum(abs(r[5] - r[2]) for r in stepping) / tt:.1f}%)")
    if still:
        mins = sum(r[6] for r in still) / 60
        lines.append(f"standing/biking: net {sum(r[3] for r in still) / mins:.1f} false steps/min "
                     f"(baseline {sum(r[5] for r in still) / mins:.1f}/min)")
    lines.append(f"mean |error| per 10 s block, all recordings: {np.mean([r[4] for r in results]):.2f} steps")
    text = "\n".join(lines)
    print(text)
    return text


def result_row(rec, models):
    t, p, e10 = score(rec, predict(models, rec["rows"]), DELAY)
    _, b, _ = score(rec, baseline_density(rec["rows"]), 0)
    return rec["name"], rec["label"], t, p, e10, b, len(rec["rows"]) / FS


def label_check(recs):
    """For marked (synthetic) data: how close the stride detector gets to the true marks."""
    lines = []
    for r in recs:
        if r["source"] == "marked" and r["label"] in step_labels.STEPPING:
            ev, _, _ = step_labels.step_events(r["rows"], r["label"])
            lines.append(f"  {r['name']}: detector {len(ev)} vs marked {len(r['events'])}")
    if lines:
        text = "Stride detector vs true marks (checks the labels used for real data):\n" + "\n".join(lines)
        print(text)
        return text
    return ""


# ---------------------------------------------------------------- export
def export(model):
    sd = {k: v.detach().numpy().astype(float) for k, v in model.state_dict().items()}
    r = lambda a: np.round(a, 6).tolist()
    return {
        "type": "step_tcn", "fs": FS, "delay": DELAY, "receptive_field": RF,
        "channels": ["ax", "ay", "az", "gx", "gy", "gz", "amag", "gmag"],
        "mean": r(sd["mean"]), "std": r(sd["std"]),
        "in_w": r(sd["inp.weight"][:, :, 0] * sd["mask"]), "in_b": r(sd["inp.bias"]),
        "blocks": [{"kernel": KERNEL, "dilation": d, "w": r(sd[f"convs.{i}.weight"]),
                    "b": r(sd[f"convs.{i}.bias"])} for i, d in enumerate(DILATIONS)],
        "out_w": r(sd["out.weight"][0, :, 0]), "out_b": float(np.round(sd["out.bias"][0], 6)),
    }


def export_ensemble(models):
    return {"type": "step_tcn_ensemble", "fs": FS, "delay": DELAY, "receptive_field": RF,
            "members": [export(m) for m in models]}


def check_export(spec, models, rows):
    counter = stepnet.StepCounter(spec)
    t0 = time.perf_counter()
    py = np.array([counter.push(tuple(s)) for s in rows])
    us = (time.perf_counter() - t0) / len(rows) * 1e6
    ref = predict(models, rows)
    return float(np.abs(py - ref).max()), abs(py.sum() - ref.sum()), us


# ---------------------------------------------------------------- main
def main():
    global WARP_MAX, ROT_MAX, PER_STRIDE, SENSOR_DROPOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "app" / "python"))
    ap.add_argument("--trim", type=float, default=5.0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--sensor-dropout", type=float, default=SENSOR_DROPOUT,
                    help="chance per training sequence of hiding all accel or all gyro channels, "
                         "so the network learns to use both")
    ap.add_argument("--members", type=int, default=MEMBERS,
                    help="networks to train and average (1 = a single network)")
    ap.add_argument("--holdout", nargs="+", metavar="FILE",
                    help="cross-validate only these recordings (quicker experiments)")
    ap.add_argument("--target", choices=["steps", "strides"], default="strides",
                    help="where the detector's steps go: one per foot, or both at each mid-swing")
    ap.add_argument("--rot", type=float, default=ROT_MAX,
                    help="max random mounting rotation in degrees (the strap never sits the same)")
    ap.add_argument("--warp", type=float, default=WARP_MAX,
                    help="max speed-up/slow-down of training crops; walking needs ~1.8 to look like running")
    ap.add_argument("--inputs", choices=list(INPUT_SETS), default="all",
                    help="hide sensors from the network, to measure what each one adds")
    ap.add_argument("--report", default=str(ROOT / "training" / "report_steps.txt"))
    args = ap.parse_args()
    WARP_MAX, ROT_MAX, PER_STRIDE = args.warp, args.rot, args.target == "strides"
    SENSOR_DROPOUT = args.sensor_dropout
    print(f"Training on {DEVICE}")

    print(f"Loading {args.data}")
    recs = load_recordings(args.data, args.trim)
    if not any(r["weights"].sum() > 0 for r in recs):
        sys.exit("No steps in any recording: record some walking or running.")
    texts = [label_check(recs)]

    if not args.no_cv:
        print(f"\nLeave-one-recording-out: {len(recs)} x {args.members} networks, "
              f"{args.iters} iterations each")
        results = []
        for i, rec in enumerate(recs):
            if args.holdout and rec["name"] not in args.holdout:
                continue
            t0 = time.time()
            models = train_ensemble(recs[:i] + recs[i + 1:], args.iters, i, args.members,
                                    inputs=args.inputs)
            results.append(result_row(rec, models))
            print(f"  [{i + 1}/{len(recs)}] {rec['name']}: true {results[-1][2]:.0f}, "
                  f"net {results[-1][3]:.1f}, baseline {results[-1][5]:.0f}  ({time.time() - t0:.0f} s)")
        texts.append(summarize(f"Step count, each recording held out (networks never saw it), "
                               f"inputs: {args.inputs}", results))

    print(f"\nTraining the final {args.members} networks on all {len(recs)} recordings")
    models = train_ensemble(recs, args.iters, 1234, args.members, log=True, inputs=args.inputs)
    texts.append(summarize("Final networks on their own training data (optimistic)",
                           [result_row(r, models) for r in recs]))
    spec = export_ensemble(models)
    longest = max(recs, key=lambda r: len(r["rows"]))
    diff, sum_diff, us = check_export(spec, models, longest["rows"])
    check = (f"Pure-Python export vs PyTorch on {longest['name']}: max density diff {diff:.2e}, "
             f"count diff {sum_diff:.4f}; {us:.0f} us/sample on this machine")
    print(check)
    if diff > 1e-3:
        sys.exit("Exported model does not match PyTorch; not writing it.")

    out = Path(args.out) / "model_steps.json"
    out.write_text(json.dumps(spec, separators=(",", ":")))
    n_params = sum(p.numel() for p in models[0].parameters())
    Path(args.report).write_text(
        f"data: {args.data}\nrecordings: {len(recs)}  model: average of {args.members} networks "
        f"of {n_params} parameters each, "
        f"receptive field {RF / FS:.2f} s, reports steps {DELAY / FS:.1f} s late\n\n"
        + "\n\n".join(t for t in texts if t) + "\n\n" + check + "\n")
    print(f"Wrote {out} and {args.report}")


if __name__ == "__main__":
    main()
