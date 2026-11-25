import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_comparisons(paths):
    cases = {}

    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        case_name = data["settings"]["arguments"]["case"]
        match = re.search(r"case(\d+)(?:_|\.|$)", case_name)
        if match is None:
            raise ValueError(f"{path}: cannot identify bus count from {case_name!r}")

        buses = int(match.group(1))
        if buses in cases:
            raise ValueError(f"Duplicate {buses}-bus result. Select one run per network.")

        methods = {row["method"]: row for row in data["results"]}
        required = {"lp_bisection", "mip_cold", "mip_lp_start"}
        missing = required - methods.keys()
        if missing:
            raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")

        for name in sorted(required):
            row = methods[name]
            elapsed = row.get("pipeline_seconds")
            if not row.get("original_soft_model_feasible"):
                raise ValueError(f"{path}: {name} has no verified feasible result")
            if elapsed is None or not np.isfinite(elapsed) or elapsed <= 0:
                raise ValueError(f"{path}: {name} has no valid pipeline time")

        cases[buses] = methods

    return dict(sorted(cases.items()))


def network_axis(ax, buses):
    ax.set_xticks(np.arange(len(buses)), buses, rotation=35, ha="right")
    ax.set_xlabel("Network size (buses)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)


def save(fig, output, name):
    path = output / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_runtime(cases, output):
    buses = list(cases)
    x = np.arange(len(buses))
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")

    series = [
        ("lp_bisection", "LP procedure", "o", "#0072B2"),
        ("mip_cold", "Cold MIP", "s", "#D55E00"),
        ("mip_lp_start", "LP procedure + warm MIP", "^", "#007C60"),
    ]

    for index, (method, label, marker, color) in enumerate(series):
        positions = x + (index - 1) * 0.22
        rows = [case[method] for case in cases.values()]
        times = np.array([row["pipeline_seconds"] for row in rows])
        time_limit = np.array([
            str(row["status"]).lower() in {"limit", "time_limit", "time limit"}
            for row in rows
        ])
        ax.scatter(positions, times, marker=marker, color=color, label=label)
        ax.scatter(
            positions[time_limit], times[time_limit],
            s=85, facecolors="none", edgecolors="#222222", linewidths=0.8,
        )

    network_axis(ax, buses)
    ax.set_yscale("log")
    ax.set_ylabel("Time (s)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False)
    save(fig, output, "r211_runtime")


def plot_gaps(cases, reference_path, output):
    reference = pd.read_csv(reference_path).set_index("buses")["lp_lower_bound"]
    if not reference.index.is_unique:
        raise ValueError(f"{reference_path}: expected one LP lower bound per network")

    buses = list(cases)
    x = np.arange(len(buses))
    lp_gaps = []
    optimum_excess = []

    for bus, case in cases.items():
        upper = case["lp_bisection"]["original_objective"]
        lower = float(reference.loc[bus])
        if not np.isfinite(lower) or upper == 0:
            raise ValueError(f"{bus} buses: invalid bound or zero denominator")
        lp_gaps.append(100 * (upper - lower) / abs(upper))

        cold = case["mip_cold"]
        if cold.get("optimal_within_solver_tolerances"):
            optimum = cold["original_objective"]
            if optimum == 0:
                raise ValueError(f"{bus} buses: MIP optimum is zero")
            optimum_excess.append(100 * (upper - optimum) / abs(optimum))
        else:
            optimum_excess.append(np.nan)

    excess = np.array(optimum_excess)
    available = np.isfinite(excess)
    fig, ax = plt.subplots(figsize=(8, 3.8), layout="constrained")
    ax.bar(x - 0.18, lp_gaps, width=0.36, color="#0072B2",
           label="Gap to reference LP bound")
    ax.bar(x[available] + 0.18, excess[available], width=0.36,
           color="#D55E00", label="Objective excess over MIP optimum")

    for position in x[~available]:
        ax.annotate("n/a", (position + 0.18, 0), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8)

    network_axis(ax, buses)
    ax.set_ylabel("Time (s)")
    ax.legend(frameon=False)
    save(fig, output, "r211_gaps")


def plot_weights(sensitivity_path, output):
    # One row per scenario. Alpha is a fraction, e.g. 0.01 for 1%.
    data = pd.read_csv(sensitivity_path)
    weights = data["selected_weight"]
    if weights.isna().any() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Every scenario row must have a finite positive selected weight")

    summary = (
        data.groupby(["buses", "alpha"])["selected_weight"]
        .agg(["min", "mean", "max"])
        .reset_index()
    )
    networks = sorted(summary["buses"].unique())
    ncols = min(3, len(networks))
    nrows = (len(networks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(8, 2.2 * nrows),
                             layout="constrained")

    for ax, bus in zip(axes.flat, networks):
        values = summary[summary["buses"] == bus]
        alpha_pct = 100 * values["alpha"].to_numpy()
        mean = values["mean"].to_numpy()
        ax.errorbar(
            alpha_pct, mean,
            yerr=[mean - values["min"].to_numpy(), values["max"].to_numpy() - mean],
            fmt="o-", color="#0072B2", capsize=3, linewidth=1,
        )
        ax.set_title(f"{bus} buses", loc="left")
        ax.set_xticks(alpha_pct)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        ax.grid(axis="y", alpha=0.25)

    for ax in list(axes.flat)[len(networks):]:
        ax.set_visible(False)

    fig.supxlabel("Demand perturbation level (%)")
    fig.supylabel("Selected weight")
    save(fig, output, "r211_weights")


def main():
    parser = argparse.ArgumentParser(description="Plot R2.11 from saved SC-DCOPF results.")
    parser.add_argument("comparisons", nargs="+", type=Path,
                        help="comparison.json files, one complete run per network")
    parser.add_argument("--reference-bounds", type=Path)
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("r211_figures"))
    args = parser.parse_args()

    plt.rcParams.update({"pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    cases = read_comparisons(args.comparisons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_runtime(cases, args.output_dir)

    if args.reference_bounds:
        plot_gaps(cases, args.reference_bounds, args.output_dir)
    else:
        print("Gap plot requires --reference-bounds.")

    if args.sensitivity:
        plot_weights(args.sensitivity, args.output_dir)
    else:
        print("Weight-sensitivity plot requires --sensitivity.")


if __name__ == "__main__":
    main()