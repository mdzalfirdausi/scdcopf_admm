"""Plot the reported SC-DCOPF demand-sensitivity weight summaries."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

# Rows give (minimum, mean, maximum) at demand perturbations of 1–5%.
WEIGHTS = {
    118: [
        (16697.78, 17110.89, 18166.03),
        (16533.05, 17221.20, 18166.03),
        (16533.05, 17213.72, 18172.93),
        (16533.05, 17205.75, 18172.93),
        (16533.05, 17139.65, 18172.94),
    ],
    162: [
        (114525.96, 134082.98, 139160.43),
        (114491.27, 132155.62, 139605.75),
        (112968.52, 130517.33, 140577.21),
        (112949.80, 129931.81, 139915.20),
        (111384.59, 129452.23, 139140.11),
    ],
    179: [
        (94592.29, 94592.29, 94592.29),
        (94592.29, 94592.29, 94592.29),
        (94592.29, 94592.29, 94592.29),
        (94592.29, 94592.29, 94592.29),
        (94592.29, 94650.93, 100456.30),
    ],
    793: [
        (17723.98, 17740.60, 17987.53),
        (17687.76, 17780.50, 18051.31),
        (17687.76, 17808.56, 18081.76),
        (17687.76, 17819.46, 18083.33),
        (17687.76, 17830.68, 18128.85),
    ],
}
PERTURBATIONS = [1, 2, 3, 4, 5]
MEAN_COLOR = "#174A73"
RANGE_COLOR = "#74A7C7"
OUTCOME_COLOR = "#AA4D19"


def main():
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(7.6, 5.5))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.85, bottom=0.23,
                        hspace=0.39, wspace=0.28)

    for ax, (buses, rows) in zip(axes.flat, WEIGHTS.items()):
        low, mean, high = zip(*([weight / 1000 for weight in row] for row in rows))
        ax.errorbar(
            PERTURBATIONS, mean,
            yerr=[[m - lo for m, lo in zip(mean, low)],
                  [hi - m for hi, m in zip(high, mean)]],
            fmt="none", ecolor=RANGE_COLOR, elinewidth=2.2, capsize=3,
        )
        ax.plot(PERTURBATIONS, mean, "o-", color=MEAN_COLOR,
                linewidth=1.7, markersize=4.5)
        if buses == 793:
            index = PERTURBATIONS.index(3)
            ax.scatter(PERTURBATIONS[index], mean[index], s=88,
                       facecolors="none", edgecolors=OUTCOME_COLOR,
                       linewidths=1.4, zorder=4)
        ax.set_title(f"{buses} buses", loc="left", fontweight="bold", pad=7)
        ax.set_xticks(PERTURBATIONS)
        ax.set_xlim(0.7, 5.3)
        ax.margins(y=0.16)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e4e9ed")

    legend = [
        Line2D([], [], color=MEAN_COLOR, marker="o", label="Mean"),
        Line2D([], [], color=RANGE_COLOR, linewidth=2.2, label="Min–max range"),
        Line2D([], [], color=OUTCOME_COLOR, marker="o", markerfacecolor="none",
               linestyle="none", label="One inconclusive search in group"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.55, 0.98),
               ncol=3, frameon=False)
    fig.supylabel(r"Selected weight ($\times 10^3$)", x=0.035, y=0.54)
    fig.supxlabel("Demand perturbation level (%)", x=0.56, y=0.13)
    fig.text(
        0.14, 0.055,
        r"$\bf{200\ buses}$   All runs reached the 40-test limit and retained "
        r"$w_f \approx 2.73\times10^{-7}$.",
        fontsize=9,
    )
    output = Path("r211_figures/r211_weights.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.13)
    plt.close(fig)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
