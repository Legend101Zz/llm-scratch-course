#!/usr/bin/env python3
"""Render the decision boundary your network learned, and the honest baseline.

The mentor wrote this. It reads CSVs your Rust writes and produces the figure
that makes the R0b claim legible to somebody who is not you.

    # the boundary under the data
    python3 capstone_r0b/render_boundary.py \
        evidence/phase0_spiral_data.csv evidence/phase0_spiral_boundary.csv

    # the baseline: the best straight line that exists on YOUR points.
    # Needs only the data file. Put the number it prints in your write-up.
    python3 capstone_r0b/render_boundary.py evidence/phase0_spiral_data.csv --best-line

Input columns:
    data      x,y,label      label is 0 or 1
    boundary  x,y,p1         p1 is the model's probability of class 1,
                             on a regular grid. 200x200 over [-1.3, 1.3]
                             is what the README asks for. Row order does
                             not matter: the grid is rebuilt from the
                             distinct x and y values.
"""

import argparse
import csv
import sys
from pathlib import Path


def read_data(path):
    xs, ys, labels = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            labels.append(int(row["label"]))
    if not xs:
        sys.exit(f"{path}: no rows")
    return xs, ys, labels


def read_grid(path):
    xs, ys, ps = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            ps.append(float(row["p1"]))
    if not xs:
        sys.exit(f"{path}: no rows")
    return xs, ys, ps


def best_straight_line(xs, ys, labels, steps=721):
    """The exact optimum over all linear classifiers, by exhaustive search.

    Every direction from 0 to pi, and every threshold along each direction.
    This is not a fitted model: it is the best a straight line can possibly do.
    """
    import math

    n = len(labels)
    best, best_angle = 0.5, 0.0
    for k in range(steps):
        a = math.pi * k / (steps - 1)
        nx, ny = math.cos(a), math.sin(a)
        proj = sorted(zip((x * nx + y * ny for x, y in zip(xs, ys)), labels))
        lab = [c for _, c in proj]
        tot0 = sum(1 for c in lab if c == 0)
        left0 = 0
        for k2 in range(n + 1):
            if k2 > 0 and lab[k2 - 1] == 0:
                left0 += 1
            right1 = (n - k2) - (tot0 - left0)
            acc = (left0 + right1) / n
            acc = max(acc, 1.0 - acc)
            if acc > best:
                best, best_angle = acc, math.degrees(a)
    return best, best_angle


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path, help="evidence/phase0_spiral_data.csv")
    ap.add_argument("boundary", type=Path, nargs="?", default=None,
                    help="evidence/phase0_spiral_boundary.csv")
    ap.add_argument("--best-line", action="store_true",
                    help="compute the best possible straight-line accuracy and stop")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    xs, ys, labels = read_data(args.data)

    if args.best_line or args.boundary is None:
        acc, angle = best_straight_line(xs, ys, labels)
        print(f"{args.data}: {len(labels)} points, "
              f"{sum(1 for c in labels if c == 0)} of class 0")
        print(f"  chance                            50.00%")
        print(f"  BEST POSSIBLE straight line       {acc * 100:.2f}%  "
              f"(normal at {angle:.1f} deg)")
        print(f"  the capstone bar                  99.00%")
        print()
        print(f"  Put the middle number in evidence/phase0_spiral.md. The gap")
        print(f"  between it and 99% is the part only a non-linear model reaches.")
        if acc > 0.80:
            print()
            print(f"  WARNING: a straight line already scores {acc * 100:.1f}%. "
                  f"Your TURN is too small,")
            print(f"  so the 99% bar is not evidence of a working non-linear model. "
                  f"README section 2.3.")
        if args.boundary is None:
            return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        sys.exit("matplotlib and numpy are missing. pip install -r requirements.txt")

    gx, gy, gp = read_grid(args.boundary)
    ux = np.array(sorted(set(gx)))
    uy = np.array(sorted(set(gy)))
    if len(ux) * len(uy) != len(gp):
        sys.exit(f"{args.boundary}: {len(gp)} rows do not form a "
                 f"{len(ux)} by {len(uy)} grid. Emit one row per grid point.")
    xi = {v: i for i, v in enumerate(ux)}
    yi = {v: i for i, v in enumerate(uy)}
    grid = np.empty((len(uy), len(ux)))
    for x, y, p in zip(gx, gy, gp):
        grid[yi[y], xi[x]] = p

    out = args.out or args.boundary.with_suffix(".png")
    fig, ax = plt.subplots(figsize=(7.2, 7))

    ax.contourf(ux, uy, grid, levels=np.linspace(0, 1, 21),
                cmap="RdBu_r", alpha=0.75)
    ax.contour(ux, uy, grid, levels=[0.5], colors="k", linewidths=1.8)

    xs_a = [x for x, c in zip(xs, labels) if c == 0]
    ys_a = [y for y, c in zip(ys, labels) if c == 0]
    xs_b = [x for x, c in zip(xs, labels) if c == 1]
    ys_b = [y for y, c in zip(ys, labels) if c == 1]
    ax.scatter(xs_a, ys_a, s=16, c="#08306b", edgecolors="white",
               linewidths=0.4, label="class 0", zorder=3)
    ax.scatter(xs_b, ys_b, s=16, c="#7f0000", edgecolors="white",
               linewidths=0.4, label="class 1", zorder=3)

    ax.set_aspect("equal")
    ax.set_title("Capstone R0b — the learned decision boundary\n"
                 "black line: p(class 1) = 0.5")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # A model that only found the linear part has a nearly straight 0.5 contour.
    # Count sign changes along the middle row as a crude "does it wrap" check.
    mid = grid[len(uy) // 2]
    flips = int(np.sum(np.diff((mid > 0.5).astype(int)) != 0))
    print(f"  the p=0.5 contour crosses the middle row {flips} times")
    if flips < 2:
        print("  WARNING: fewer than 2 crossings means the boundary is "
              "essentially a straight line.")
        print("  A two-turn spiral needs a boundary that wraps. "
              "README section 6.2, curve C.")


if __name__ == "__main__":
    main()
