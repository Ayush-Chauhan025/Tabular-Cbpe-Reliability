# Draw workflow

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def build(output_path="results/column_outage_v2_calibration/figures/column_outage_workflow.png"):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    navy = "#1f4e79"
    pale = "#eaf1f5"
    orange = "#d47d35"
    green = "#287c76"

    def box(x, y, width, height, title, body, edge=navy, fill=pale):
        patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.12,rounding_size=0.14",
            linewidth=1.8, edgecolor=edge, facecolor=fill)
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.68, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#17324d")
        ax.text(x + width / 2, y + height * 0.35, body, ha="center", va="center",
            fontsize=9, color="#1b2935", wrap=True)

    def arrow(start, end, color=navy):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
            linewidth=1.7, color=color, connectionstyle="arc3,rad=0"))

    box(0.25, 4.05, 2.3, 1.35, "Data and splits", "Train / calibration / test\n10 seeds or 4 time windows")
    box(3.1, 4.05, 2.3, 1.35, "Fit and calibrate", "Train-only preprocessing\nClassifier + sigmoid calibration")
    box(5.95, 4.05, 2.45, 1.35, "Freeze the pipeline", "Baseline predictions\nModel and calibrator stay fixed")
    box(9.0, 4.05, 2.7, 1.35, "Sample outages", "10%, 25%, 50% of columns\nUp to 20 unique groups")
    box(1.0, 1.45, 3.1, 1.55, "Drop and restore columns", "Remove selected test columns\nSchema adapter inserts missing values", edge=orange, fill="#fbf2e9")
    box(4.55, 1.45, 3.1, 1.55, "Impute and predict", "Training-fitted median / mode\nFrozen classifier outputs probabilities")
    box(8.1, 1.45, 3.2, 1.55, "Compare monitor and truth", "CBPE: accuracy, precision, recall, F1\nLabels used offline for evaluation", edge=green, fill="#eaf3f0")

    arrow((2.55, 4.72), (3.05, 4.72))
    arrow((5.4, 4.72), (5.9, 4.72))
    arrow((8.4, 4.72), (8.95, 4.72))
    arrow((10.3, 4.0), (3.0, 3.1), orange)
    arrow((4.1, 2.22), (4.5, 2.22))
    arrow((7.7, 2.22), (8.05, 2.22))
    arrow((7.15, 4.0), (9.5, 3.1), green)

    fig.suptitle("Column-outage evaluation workflow", fontsize=16, fontweight="bold", color="#17324d", y=0.96)
    fig.tight_layout(pad=1.1)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    print(build())
