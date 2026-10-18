# Summarize Brier score and equal-width ECE after column outages.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ("credit", "heart", "miniboone", "electricity")
DISPLAY = {
    "credit": "Credit",
    "heart": "Heart Disease",
    "miniboone": "MiniBooNE",
    "electricity": "Electricity"
}
METRICS = ("brier_score", "ece_10_equal_width")


def summarize(results_dir="results/column_outage_v2_calibration"):
    results_dir = Path(results_dir)
    runs_path = results_dir / "column_outage_runs.csv.gz"
    if not runs_path.exists():
        runs_path = results_dir / "column_outage_runs.csv"
    runs = pd.read_csv(runs_path)
    if not {"brier_score", "ece_10_equal_width", "baseline_brier_score", "baseline_ece_10_equal_width"}.issubset(runs.columns):
        raise ValueError("Run file is missing one or more probability-quality columns")

    split_keys = ["dataset", "model", "split_seed", "fold_index", "outage_fraction_requested", "n_absent_columns"]
    aggregations = {metric: (metric, "mean") for metric in METRICS}
    aggregations.update({f"baseline_{metric}": (f"baseline_{metric}", "first") for metric in METRICS})
    split_means = runs.groupby(split_keys, as_index=False).agg(**aggregations)
    for metric in METRICS:
        split_means[f"{metric}_change"] = split_means[metric] - split_means[f"baseline_{metric}"]
    split_means.to_csv(results_dir / "calibration_quality_split_means.csv", index=False)

    summary_keys = ["dataset", "outage_fraction_requested", "n_absent_columns"]
    columns = [*METRICS, *(f"baseline_{metric}" for metric in METRICS), *(f"{metric}_change" for metric in METRICS)]
    summary = split_means.groupby(summary_keys, as_index=False)[columns].mean()
    summary.to_csv(results_dir / "calibration_quality_by_outage.csv", index=False)

    baseline = summary.groupby("dataset", as_index=False)[[f"baseline_{metric}" for metric in METRICS]].first()
    baseline["outage_fraction_requested"] = 0.0
    baseline["n_absent_columns"] = 0
    for metric in METRICS:
        baseline[metric] = baseline[f"baseline_{metric}"]
        baseline[f"{metric}_change"] = 0.0
    plot_data = pd.concat([baseline[summary.columns], summary], ignore_index=True).sort_values(["dataset", "outage_fraction_requested"])
    plot_data.to_csv(results_dir / "calibration_quality_plot_data.csv", index=False)

    _write_report(summary, results_dir)
    _make_figure(plot_data, results_dir)
    return summary


def _write_report(summary, results_dir):
    lines = [
        "# Probability quality under test-time column outage",
        "",
        "Brier score is the mean squared error between positive-class probabilities and the observed labels. ECE uses 10 equal-width bins to summarize calibration; its value depends on the binning scheme. Test labels are used only to calculate these offline scores and are never passed to CBPE.",
        "",
        "The table averages outage groups within each model and split, then weights model/split outcomes equally. The baseline is paired by the same model/split. Requested outage fractions are rounded up to whole columns; report the number removed and realized share when comparing datasets.",
        "",
        "| Dataset | Requested outage | Columns removed | Brier score | Change vs baseline | 10-bin ECE | Change vs baseline |",
        "|---|---:|---:|---:|---:|---:|---:|"
    ]
    for dataset in DATASETS:
        base_rows = summary[summary["dataset"] == dataset]
        if base_rows.empty:
            continue
        base_brier = float(base_rows.iloc[0]["baseline_brier_score"])
        base_ece = float(base_rows.iloc[0]["baseline_ece_10_equal_width"])
        lines.append(f"| {DISPLAY[dataset]} | 0% | 0 | {base_brier:.4f} | 0.0000 | {base_ece:.4f} | 0.0000 |")
        for _, row in base_rows.sort_values("outage_fraction_requested").iterrows():
            requested = float(row["outage_fraction_requested"])
            brier_change = float(row["brier_score"] - row["baseline_brier_score"])
            ece_change = float(row["ece_10_equal_width"] - row["baseline_ece_10_equal_width"])
            lines.append(
                f"| {DISPLAY[dataset]} | {requested:.0%} | {int(row['n_absent_columns'])} | "
                f"{row['brier_score']:.4f} | {brier_change:+.4f} | {row['ece_10_equal_width']:.4f} | {ece_change:+.4f} |"
            )
    lines.extend([
        "",
        "The Brier score reflects both reliability and resolution, so it is not a pure calibration-error measure. ECE is sensitive to the selected binning scheme and sample size. These results describe the tested held-out conditions; they do not establish that calibration change caused CBPE estimation error.",
        ""
    ])
    (results_dir / "calibration_quality_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _make_figure(plot_data, results_dir):
    colors = {"credit": "#1f77b4", "heart": "#ff7f0e", "miniboone": "#2ca02c", "electricity": "#d62728"}
    x_values = [0, 10, 25, 50]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, metric, title in zip(axes, METRICS, ("Brier score", "10-bin ECE")):
        for dataset in DATASETS:
            rows = plot_data[plot_data["dataset"] == dataset].sort_values("outage_fraction_requested")
            if rows.empty:
                continue
            ax.plot(x_values, rows[metric].to_numpy(dtype=float), marker="o", linewidth=2,
                color=colors[dataset], label=DISPLAY[dataset])
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x_values)
        ax.set_xlabel("Requested feature outage (%)")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Probability quality by feature-outage severity", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    fig.savefig(results_dir / "calibration_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    summarize()
