import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "table_C_collected_results.csv"

    buses = [118, 162, 179, 200, 240, 300, 500, 588, 793, 1888, 1951, 2000]

    # Historical cold-MIP totals from the replacement runs with a 1800 s limit.
    cold_replacement = {
        240: 1806.02,
        793: 1808.69,
        1888: 1871.32,
        1951: 1895.29,
    }

    # Historical totals including both the LP search and the initialized MIP.
    warm_total = {
        118: 22.30,
        162: 22.12,
        179: 31.76,
        200: 31.16,
        240: 2392.55,
        300: 256.22,
        500: 1266.35,
        588: 379.59,
        793: 2217.94,
        1888: 2723.55,
        1951: 3426.04,
        2000: 1514.60,
    }

    rows = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            prefix = row["case"].strip().split("_", 1)[0]
            if not prefix.isdigit() or int(prefix) not in buses:
                continue
            n = int(prefix)
            if n in rows:
                raise ValueError(f"Duplicate {n}-bus row in {csv_path.name}")
            rows[n] = row

    missing = sorted(set(buses) - rows.keys())
    if missing:
        raise ValueError(f"Missing network sizes in {csv_path.name}: {missing}")

    lp = [float(rows[n]["LP_total_seconds"]) for n in buses]
    cold = [
        cold_replacement[n] if n in cold_replacement
        else float(rows[n]["MIP_total_seconds"])
        for n in buses
    ]
    warm = [warm_total[n] for n in buses]
    time_limited = [i for i, n in enumerate(buses) if n in cold_replacement]

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10})
    fig, ax = plt.subplots(figsize=(9.2, 4.1), layout="constrained")

    series = [
        (-0.22, lp, "o", "#0072B2", "LP candidate construction", []),
        (0.0, cold, "s", "#D55E00", "Cold MIP", time_limited),
        (0.22, warm, "^", "#007C60", "LP search + initialized MIP", time_limited),
    ]

    for offset, times, marker, color, label, ring_indices in series:
        positions = [i + offset for i in range(len(buses))]
        ax.scatter(positions, times, marker=marker, color=color,
                   s=34, label=label, zorder=3)
        if ring_indices:
            ax.scatter(
                [positions[i] for i in ring_indices],
                [times[i] for i in ring_indices],
                s=110, facecolors="none", edgecolors="#222222",
                linewidths=0.8, zorder=4,
            )

    ax.set_yscale("log")
    ax.set_xticks(range(len(buses)), [str(n) for n in buses],
                  rotation=35, ha="right")
    ax.set_xlabel("Network size (buses)")
    ax.set_ylabel("Time (s)")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False)

    output = project_dir / "r211_figures" / "r211_runtime_time_label.pdf"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()