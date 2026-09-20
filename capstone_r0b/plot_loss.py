#!/usr/bin/env python3
"""Plot the capstone loss curve.

The mentor wrote this. It reads the CSV your Rust writes and produces the
figure that goes in `evidence/`. It never touches the Rust crate.

    python3 capstone_r0b/plot_loss.py evidence/phase0_spiral_loss.csv

Input columns: epoch,loss,train_accuracy

The loss panel uses a log y axis on purpose. Phase 4 of the curve spans 0.03
down to 0.002, and that whole range is a flat line against a linear axis.
"""

import argparse
import csv
import math
import sys
from pathlib import Path


def read_curve(path):
    epochs, losses, accs = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            accs.append(float(row["train_accuracy"]))
    if not epochs:
        sys.exit(f"{path}: no rows. Did the run write its header only?")
    return epochs, losses, accs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="evidence/phase0_spiral_loss.csv")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output PNG (default: alongside the CSV)")
    args = ap.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is missing. pip install -r requirements.txt")

    epochs, losses, accs = read_curve(args.csv)
    out = args.out or args.csv.with_suffix(".png")

    ln2 = math.log(2.0)
    hit = next((e for e, a in zip(epochs, accs) if a > 0.99), None)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(epochs, losses, lw=1.6, color="#8b3a1a", label="train loss")
    ax1.axhline(ln2, ls="--", lw=1.1, color="#1f6f43",
                label=f"ln(2) = {ln2:.4f}, the no-information loss")
    if hit is not None:
        ax1.axvline(hit, ls=":", lw=1.1, color="#b8860b",
                    label=f"99% accuracy at epoch {hit}")
    ax1.set_yscale("log")
    ax1.set_ylabel("cross-entropy (log scale)")
    ax1.set_title(f"Capstone R0b — two spirals   ({args.csv.name})")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(loc="upper right", fontsize=9)

    ax2.plot(epochs, [a * 100 for a in accs], lw=1.6, color="#1f6f43")
    ax2.axhline(99.0, ls="--", lw=1.1, color="#b8860b", label="the 99% bar")
    ax2.axhline(60.5, ls=":", lw=1.1, color="#666666",
                label="best possible straight line (60.5%)")
    ax2.axhline(50.0, ls=":", lw=1.0, color="#999999", label="chance (50%)")
    ax2.set_ylim(45, 102)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("train accuracy (%)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=140)

    print(f"wrote {out}")
    print(f"  epochs           {epochs[0]} to {epochs[-1]}  ({len(epochs)} rows)")
    print(f"  step-0 loss      {losses[0]:.6f}   (ln 2 = {ln2:.6f})")
    print(f"  final loss       {losses[-1]:.6f}")
    print(f"  best accuracy    {max(accs) * 100:.2f}%")
    print(f"  reached 99%      {'epoch ' + str(hit) if hit is not None else 'NEVER'}")
    if abs(losses[0] - ln2) > 0.25:
        print("  WARNING: the step-0 loss is far from ln(2). "
              "That is a forward-pass or init fault. See README section 7, rung 1.")
    if max(accs) <= 0.99:
        print("  WARNING: the 99% bar was not reached. "
              "Work the diagnostic ladder in README section 7, from rung 0.")


if __name__ == "__main__":
    main()
