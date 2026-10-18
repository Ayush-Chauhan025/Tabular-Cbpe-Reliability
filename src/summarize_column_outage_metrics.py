# Summarize CBPE accuracy, precision, recall, and F1 under column outages.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ("accuracy", "precision", "recall", "f1")
DATASETS = ("credit", "heart", "miniboone", "electricity")
DISPLAY = {"credit": "Credit", "heart": "Heart Disease", "miniboone": "MiniBooNE", "electricity": "Electricity"}


def _columns(metric):
    if metric == "accuracy":
        return {
            "actual": "actual_accuracy",
            "estimate": "cbpe_accuracy_estimate",
            "signed": "signed_estimation_error",
            "absolute": "absolute_estimation_error",
            "coverage": "accuracy_interval_covers_actual",
            "actual_change": "actual_accuracy_change",
            "signed_change": "signed_error_change",
            "absolute_change": "absolute_error_change"
        }
    return {
        "actual": f"actual_{metric}",
        "estimate": f"cbpe_{metric}_estimate",
        "signed": f"{metric}_signed_estimation_error",
        "absolute": f"{metric}_absolute_estimation_error",
        "coverage": f"{metric}_interval_covers_actual",
        "actual_change": f"actual_{metric}_change",
        "signed_change": f"{metric}_signed_error_change",
        "absolute_change": f"{metric}_absolute_error_change"
    }


def summarize(results_dir="results/column_outage_v2_calibration"):
    results_dir = Path(results_dir)
    runs_path = results_dir / "column_outage_runs.csv.gz"
    if not runs_path.exists():
        runs_path = results_dir / "column_outage_runs.csv"
    runs = pd.read_csv(runs_path)
    if not runs["schema_restoration_matches_all_missing_control"].all():
        raise ValueError("At least one absent-column input differs from the all-missing control")

    split_group = ["dataset", "model", "split_seed", "fold_index", "outage_fraction_requested", "n_absent_columns"]
    aggregations = {"n_feature_groups": ("replicate", "size")}
    for metric in METRICS:
        columns = _columns(metric)
        for name in ("actual", "estimate", "signed", "absolute", "coverage", "actual_change", "absolute_change"):
            aggregations[f"{metric}_{name}"] = (columns[name], "mean")
        aggregations[f"baseline_{metric}_actual"] = (f"baseline_{columns['actual']}", "mean")
        aggregations[f"baseline_{metric}_estimate"] = (f"baseline_{columns['estimate']}", "mean")
        aggregations[f"baseline_{metric}_absolute"] = (f"baseline_{columns['absolute']}", "mean")
        aggregations[f"baseline_{metric}_coverage"] = (f"baseline_{columns['coverage']}", "mean")
    split_means = runs.groupby(split_group, as_index=False).agg(**aggregations)
    split_means.to_csv(results_dir / "column_outage_all_metric_split_means.csv", index=False)

    summary_group = ["dataset", "outage_fraction_requested", "n_absent_columns"]
    summary_aggregations = {"n_model_split_outcomes": ("fold_index", "size"), "n_feature_groups": ("n_feature_groups", "sum")}
    for metric in METRICS:
        for name in ("actual", "estimate", "signed", "absolute", "coverage", "actual_change", "absolute_change"):
            summary_aggregations[f"{metric}_{name}"] = (f"{metric}_{name}", "mean")
        for name in ("actual", "estimate", "absolute", "coverage"):
            summary_aggregations[f"baseline_{metric}_{name}"] = (f"baseline_{metric}_{name}", "mean")
    macro = split_means.groupby(summary_group, as_index=False).agg(**summary_aggregations)
    macro.to_csv(results_dir / "column_outage_all_metric_summary.csv", index=False)

    mask_aggregations = {"n_mask_split_model_conditions": ("replicate", "size")}
    for metric in METRICS:
        columns = _columns(metric)
        mask_aggregations[f"{metric}_actual_change_mean"] = (columns["actual_change"], "mean")
        mask_aggregations[f"{metric}_actual_change_median"] = (columns["actual_change"], "median")
        mask_aggregations[f"{metric}_absolute_error_change_mean"] = (columns["absolute_change"], "mean")
        mask_aggregations[f"{metric}_absolute_error_worsened_fraction"] = (columns["absolute_change"], lambda values: (values > 0).mean())
    mask_summary = runs.groupby(["dataset", "outage_fraction_requested", "n_absent_columns"], as_index=False).agg(**mask_aggregations)
    mask_summary.to_csv(results_dir / "column_outage_all_metric_mask_summary.csv", index=False)

    bootstrap = _bootstrap_by_seed(split_means)
    bootstrap.to_csv(results_dir / "column_outage_all_metric_seed_bootstrap.csv", index=False)
    _write_report(macro, bootstrap, results_dir)
    _make_accuracy_figure(macro, results_dir)
    _make_figure(macro, results_dir)
    return macro


def _bootstrap_by_seed(split_means, n_bootstrap=10000, seed=20261017):
    split_means = split_means.copy()
    split_means["resampling_unit"] = np.where(
        split_means["dataset"] == "electricity",
        "window_" + split_means["fold_index"].astype(str),
        "seed_" + split_means["split_seed"].astype(str)
    )
    aggregations = {}
    for metric in METRICS:
        aggregations[f"{metric}_actual_change"] = (f"{metric}_actual_change", "mean")
        aggregations[f"{metric}_absolute_change"] = (f"{metric}_absolute_change", "mean")
        aggregations[f"{metric}_coverage"] = (f"{metric}_coverage", "mean")
    units = split_means.groupby(["dataset", "resampling_unit", "outage_fraction_requested"], as_index=False).agg(**aggregations)
    rng = np.random.default_rng(seed)
    rows = []
    for (dataset, fraction), group in units.groupby(["dataset", "outage_fraction_requested"]):
        row = {
            "dataset": dataset,
            "outage_fraction_requested": fraction,
            "n_split_units": len(group),
            "unit_type": "chronological window" if dataset == "electricity" else "split seed"
        }
        for metric in METRICS:
            for measure in ("actual_change", "absolute_change", "coverage"):
                values = group[f"{metric}_{measure}"].to_numpy(dtype=float)
                samples = values[rng.integers(0, len(values), size=(n_bootstrap, len(values)))].mean(axis=1)
                row[f"{metric}_{measure}_mean"] = values.mean()
                row[f"{metric}_{measure}_ci_low"] = np.quantile(samples, 0.025)
                row[f"{metric}_{measure}_ci_high"] = np.quantile(samples, 0.975)
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(macro, bootstrap, results_dir):
    lines = [
        "# CBPE estimates for four binary-classification metrics",
        "",
        "The report covers accuracy, positive-class precision, positive-class recall, and positive-class F1. CBPE estimates use test predictions and calibrated class probabilities; test labels are used only for the paired observed metrics and empirical interval-coverage assessment.",
        "",
        "All summaries average feature groups within each model/split, then average model/split outcomes equally. Seed bootstrap ranges average folds and models within each of 10 split seeds for Credit, Heart Disease, and MiniBooNE. Electricity has four windows from one series, so its resampled ranges are descriptive. Group-level and seed-level summaries are not independent of model fitting or the chosen datasets.",
        "",
        "## Metric results at 50% column outage",
        "",
        "Absolute errors and changes are percentage points. Coverage is the empirical share of nominal 95% normal/delta CBPE intervals that contained the observed metric.",
        "",
        "| Dataset | Metric | Actual change from baseline | CBPE estimate | Absolute error | Change in absolute error | Interval coverage |",
        "|---|---|---:|---:|---:|---:|---:|"
    ]
    for dataset in DATASETS:
        rows = macro[(macro["dataset"] == dataset) & (macro["outage_fraction_requested"] == 0.50)]
        if rows.empty:
            continue
        row = rows.iloc[0]
        for metric in METRICS:
            lines.append(
                f"| {DISPLAY[dataset]} | {metric.title()} | {row[f'{metric}_actual_change']*100:+.2f} | "
                f"{row[f'{metric}_estimate']*100:.2f} | {row[f'{metric}_absolute']*100:.2f} | "
                f"{row[f'{metric}_absolute_change']*100:+.2f} | {row[f'{metric}_coverage']*100:.1f}% |"
            )

    lines.extend([
        "",
        "## Paired seed-level changes by outage severity",
        "",
        "The intervals below are paired percentile bootstrap ranges over split seeds after averaging folds and models within each seed. They describe variability across these split/training-seed configurations and do not include feature-mask sampling uncertainty. Electricity ranges resample four chronological windows and must not be interpreted as population-level confidence intervals.",
        "",
        "| Dataset | Outage | Metric | Actual change (pp, 95% range) | Absolute-error change (pp, 95% range) | Coverage |",
        "|---|---:|---|---:|---:|---:|"
    ])
    for dataset in DATASETS:
        rows = bootstrap[bootstrap["dataset"] == dataset].sort_values("outage_fraction_requested")
        for _, row in rows.iterrows():
            for metric in METRICS:
                lines.append(
                    f"| {DISPLAY[dataset]} | {row.outage_fraction_requested*100:.0f}% | {metric.title()} | "
                    f"{row[f'{metric}_actual_change_mean']*100:+.2f} [{row[f'{metric}_actual_change_ci_low']*100:+.2f}, {row[f'{metric}_actual_change_ci_high']*100:+.2f}] | "
                    f"{row[f'{metric}_absolute_change_mean']*100:+.2f} [{row[f'{metric}_absolute_change_ci_low']*100:+.2f}, {row[f'{metric}_absolute_change_ci_high']*100:+.2f}] | "
                    f"{row[f'{metric}_coverage_mean']*100:.1f}% |"
                )
    lines.extend([
        "",
        "Interpretation should account for baseline coverage and dataset context. For example, a low outage-condition interval coverage can reflect poor baseline coverage as well as the feature-outage condition. A positive change in absolute error indicates that the estimator moved farther from the observed metric than at the paired baseline.",
        ""
    ])
    (results_dir / "column_outage_all_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def _make_figure(macro, results_dir):
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    colors = {"credit": "#1f77b4", "heart": "#ff7f0e", "miniboone": "#2ca02c", "electricity": "#d62728"}
    fractions = [0, 10, 25, 50]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for dataset in DATASETS:
            rows = macro[(macro["dataset"] == dataset)].sort_values("outage_fraction_requested")
            if rows.empty:
                continue
            baseline = rows.iloc[0][f"baseline_{metric}_absolute"]
            values = [baseline] + rows[f"{metric}_absolute"].tolist()
            ax.plot(fractions, np.asarray(values) * 100, marker="o", linewidth=2, color=colors[dataset], label=DISPLAY[dataset])
        ax.set_title(metric.title(), fontweight="bold")
        ax.set_xticks(fractions)
        ax.set_xlabel("Input columns unavailable (%)")
        ax.set_ylabel("Absolute estimation error (pp)")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("CBPE metric-estimation error by outage severity", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(figure_dir / "all_metric_absolute_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_accuracy_figure(macro, results_dir):
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    display = {
        "credit": "Credit",
        "heart": "Heart Disease",
        "miniboone": "MiniBooNE",
        "electricity": "Electricity"
    }
    fractions = [0, 10, 25, 50]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)

    for ax, dataset in zip(axes.flat, DATASETS):
        rows = macro[macro["dataset"] == dataset].sort_values("outage_fraction_requested")
        if rows.empty:
            ax.set_visible(False)
            continue

        baseline_actual = rows.iloc[0]["baseline_accuracy_actual"]
        baseline_estimate = rows.iloc[0]["baseline_accuracy_estimate"]
        actual = [baseline_actual, *rows["accuracy_actual"].tolist()]
        estimate = [baseline_estimate, *rows["accuracy_estimate"].tolist()]
        ax.plot(fractions, np.asarray(actual) * 100, marker="o", linewidth=2,
            color="#1f77b4", label="Observed accuracy")
        ax.plot(fractions, np.asarray(estimate) * 100, marker="s", linewidth=2,
            color="#ff7f0e", label="CBPE estimate")
        ax.set_title(display[dataset], fontweight="bold")
        ax.set_xticks(fractions)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Requested outage (%)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.25)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Observed and CBPE-estimated accuracy under column outages",
        fontsize=14, fontweight="bold")
    fig.text(0.5, 0.035,
        "Models and calibrators remain frozen; missing columns are restored and imputed with training-fitted values.",
        ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    fig.savefig(figure_dir / "accuracy_estimates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    summarize()
