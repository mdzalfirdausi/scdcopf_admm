"""Plot the two objective-gap measures for the 12 baseline SC-DCOPF cases."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Table 2's gaps to the reference LP dual bound, at the reported precision.
# The cold-MIP bound in table_C_collected_results.csv is a different bound.
LP_BOUND_GAP_PCT = {
    118: 35.80,
    162: 13.49,
    179: 21.58,
    200: 0.00,
    240: 32.33,
    300: 3.19,
    500: 0.76,
    588: 0.06,
    793: 3.12,
    1888: 0.56,
    1951: 0.77,
    2000: 5.00,
}
OPTIMAL_CASES = {118, 162, 179, 200, 300, 500, 588, 2000}


def read_mip_excess(path):
    excess = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            buses = int(row["case"].strip().split("_", 1)[0])
            if buses not in OPTIMAL_CASES:
                continue
            if buses in excess:
                raise ValueError(f"{path}: duplicate row for {buses} buses")
            if (
                row["MIP_status"].strip().lower() != "optimal"
                or row["MIP_verified"].strip().lower() != "true"
            ):
                raise ValueError(f"{path}: {buses}-bus MIP optimum is not verified")

            gap = float(row["Delta_pct"])
            if not math.isfinite(gap) or gap < -0.005:
                raise ValueError(f"{path}: invalid objective excess for {buses} buses")
            # The baseline primal objectives in this CSV are rounded.
            excess[buses] = max(0.0, gap)

    missing = sorted(OPTIMAL_CASES - excess.keys())
    if missing:
        raise ValueError(f"{path}: missing optimal MIP rows for bus counts {missing}")
    return excess


def draw_gaps(mip_excess, output):
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(10, 4.3), layout="constrained")
    networks = sorted(LP_BOUND_GAP_PCT)
    positions = list(range(len(networks)))
    width = 0.33
    blue = "#236FA8"
    orange = "#C95D00"

    ax.bar(
        [x - width / 2 for x in positions],
        [LP_BOUND_GAP_PCT[buses] for buses in networks],
        width=width, color=blue, label="Gap to reference LP bound",
    )
    ax.bar(
        [x + width / 2 for x, buses in enumerate(networks) if buses in mip_excess],
        [mip_excess[buses] for buses in networks if buses in mip_excess],
        width=width, color=orange, label="Relative gap to MIP optimum",
    )

    for x, buses in enumerate(networks):
        if buses not in mip_excess:
            ax.annotate(
                "n/a", (x + width / 2, 0), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom", fontsize=10,
            )

    ax.set_xticks(positions, networks, rotation=35, ha="right")
    ax.set_xlabel("Network size (buses)")
    ax.set_ylabel("Gap (%)")
    ax.set_ylim(0, 39)
    ax.set_yticks(range(0, 36, 5))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E8E8E8")
    ax.legend(loc="upper right", frameon=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("table_C_collected_results.csv"),
        help="baseline cold-MIP results CSV from the repository",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("r211_figures/r211_gaps.pdf"),
        help="output figure path",
    )
    args = parser.parse_args()
    try:
        draw_gaps(read_mip_excess(args.csv), args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
