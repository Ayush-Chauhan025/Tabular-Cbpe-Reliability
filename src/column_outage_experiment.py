# Evaluate CBPE after required test-time feature columns disappear.

from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from calibration import calibrate_model
from data import get_tabular_splits
from estimation import estimate_cbpe_metrics
from preprocessing import preprocess_tabular_data


def _align_absent_columns(x_request, expected_columns, absent_columns):
    absent_columns = set(absent_columns)
    unexpected = set(x_request.columns) - set(expected_columns)
    if unexpected:
        raise ValueError(f"Unexpected input columns: {sorted(unexpected)}")
    if absent_columns & set(x_request.columns):
        raise ValueError("The request must not contain columns declared absent")
    if set(expected_columns) - set(x_request.columns) != absent_columns:
        raise ValueError("Declared absent columns must match columns missing from the request")
    return x_request.reindex(columns=expected_columns)


def _metric_record(y_true, predictions, probabilities, classes, positive_label):
    estimates = estimate_cbpe_metrics(
        probabilities=probabilities,
        predictions=predictions,
        classes=classes,
        positive_label=positive_label,
        confidence_level=0.95
    )
    actual = float(np.mean(np.asarray(predictions) == np.asarray(y_true)))
    estimated = estimates["accuracy_estimate"]
    actual_metrics = {
        "precision": float(precision_score(y_true, predictions, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=positive_label, zero_division=0))
    }
    other_metric_records = {}
    for metric, metric_actual in actual_metrics.items():
        metric_estimate = estimates[f"{metric}_estimate"]
        other_metric_records.update({
            f"actual_{metric}": metric_actual,
            f"cbpe_{metric}_estimate": metric_estimate,
            f"{metric}_signed_estimation_error": metric_estimate - metric_actual,
            f"{metric}_absolute_estimation_error": abs(metric_estimate - metric_actual),
            f"{metric}_interval_lower": estimates[f"{metric}_ci_lower"],
            f"{metric}_interval_upper": estimates[f"{metric}_ci_upper"],
            f"{metric}_interval_covers_actual": estimates[f"{metric}_ci_lower"] <= metric_actual <= estimates[f"{metric}_ci_upper"]
        })
    positive_index = int(np.where(np.asarray(classes) == positive_label)[0][0])
    positive_scores = np.asarray(probabilities, dtype=float)[:, positive_index]
    binary_labels = (np.asarray(y_true) == positive_label).astype(float)
    bin_indices = np.minimum(np.digitize(positive_scores, np.linspace(0.0, 1.0, 11)[1:-1]), 9)
    ece_10 = 0.0
    for bin_index in range(10):
        in_bin = bin_indices == bin_index
        if in_bin.any():
            ece_10 += float(in_bin.mean()) * abs(float(binary_labels[in_bin].mean() - positive_scores[in_bin].mean()))
    return {
        "actual_accuracy": actual,
        "brier_score": float(np.mean((positive_scores - binary_labels) ** 2)),
        "ece_10_equal_width": ece_10,
        "cbpe_accuracy_estimate": estimated,
        "signed_estimation_error": estimated - actual,
        "absolute_estimation_error": abs(estimated - actual),
        "accuracy_interval_lower": estimates["accuracy_ci_lower"],
        "accuracy_interval_upper": estimates["accuracy_ci_upper"],
        "accuracy_interval_covers_actual": estimates["accuracy_ci_lower"] <= actual <= estimates["accuracy_ci_upper"],
        **other_metric_records
    }


def _sample_feature_groups(columns, fractions, n_groups, seed):
    rng = np.random.default_rng(seed)
    n_features = len(columns)
    for fraction in fractions:
        group_size = min(n_features, max(1, int(np.ceil(fraction * n_features))))
        n_possible_groups = comb(n_features, group_size)
        n_groups_to_sample = min(n_groups, n_possible_groups)
        if n_groups_to_sample == n_possible_groups:
            sampled_groups = list(combinations(columns, group_size))
        else:
            unique_groups = set()
            while len(unique_groups) < n_groups_to_sample:
                selected = tuple(sorted(rng.choice(columns, size=group_size, replace=False).tolist()))
                unique_groups.add(selected)
            sampled_groups = sorted(unique_groups)
        for replicate, selected in enumerate(sampled_groups):
            yield fraction, group_size, replicate, selected


def run_column_outage_experiment(dataset_configs, model_factories, split_seeds, n_folds,
    outage_fractions, groups_per_split_fraction, results_dir, checkpoint_every=10):

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = results_dir / "column_outage_runs.partial.csv"
    output_rows = []
    completed_models = 0

    for dataset_config in dataset_configs:
        dataset = dataset_config["name"]
        positive_label = dataset_config["positive_label"]
        dataset_seeds = split_seeds[:1] if dataset_config.get("split_strategy") == "chronological" else split_seeds
        for split_seed in dataset_seeds:
            splits = get_tabular_splits(dataset_config, random_state=split_seed, n_folds=n_folds)
            for fold_index, (x_train, y_train, x_test, y_test, x_cal, y_cal) in enumerate(splits):
                preprocessor, x_train_processed = preprocess_tabular_data(x_train)
                x_cal_processed = preprocessor.transform(x_cal)
                x_test_processed = preprocessor.transform(x_test)
                columns = list(x_train.columns)
                groups = list(_sample_feature_groups(
                    columns,
                    outage_fractions,
                    groups_per_split_fraction,
                    seed=split_seed * 1009 + fold_index
                ))

                for model_name, model_factory in model_factories.items():
                    print(f"Fitting {dataset}; model={model_name}; seed={split_seed}; fold={fold_index + 1}/{len(splits)}")
                    model = model_factory(split_seed + fold_index)
                    model.fit(x_train_processed, y_train)
                    calibrated_model = calibrate_model(model, x_cal_processed, y_cal)
                    baseline_probabilities = calibrated_model.predict_proba(x_test_processed)
                    baseline_predictions = calibrated_model.classes_[baseline_probabilities.argmax(axis=1)]
                    baseline_metrics = _metric_record(
                        y_test, baseline_predictions, baseline_probabilities,
                        calibrated_model.classes_, positive_label
                    )

                    for fraction, group_size, replicate, absent_columns in groups:
                        request = x_test.drop(columns=list(absent_columns))
                        aligned_test = _align_absent_columns(request, columns, absent_columns)
                        all_missing_control = x_test.copy()
                        for column in absent_columns:
                            all_missing_control[column] = np.nan
                        if not aligned_test.equals(all_missing_control):
                            raise AssertionError("Schema restoration differs from the all-missing-value control")
                        aligned_processed = preprocessor.transform(aligned_test)
                        probabilities = calibrated_model.predict_proba(aligned_processed)
                        predictions = calibrated_model.classes_[probabilities.argmax(axis=1)]
                        outage_metrics = _metric_record(
                            y_test, predictions, probabilities,
                            calibrated_model.classes_, positive_label
                        )
                        output_rows.append({
                            "dataset": dataset,
                            "model": model_name,
                            "split_seed": split_seed,
                            "fold_index": fold_index,
                            "outage_fraction_requested": fraction,
                            "outage_fraction_realized": group_size / len(columns),
                            "n_input_features": len(columns),
                            "n_absent_columns": group_size,
                            "replicate": replicate,
                            "absent_columns": ", ".join(absent_columns),
                            "schema_adapter_inserted_all_missing": True,
                            "schema_restoration_matches_all_missing_control": True,
                            "unadapted_request_missing_required_columns": True,
                            "accuracy_interval_method": "normal_delta_approximation",
                            **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                            **outage_metrics,
                            "actual_accuracy_change": outage_metrics["actual_accuracy"] - baseline_metrics["actual_accuracy"],
                            "signed_error_change": outage_metrics["signed_estimation_error"] - baseline_metrics["signed_estimation_error"],
                            "absolute_error_change": outage_metrics["absolute_estimation_error"] - baseline_metrics["absolute_estimation_error"],
                            **{f"actual_{metric}_change": outage_metrics[f"actual_{metric}"] - baseline_metrics[f"actual_{metric}"] for metric in ("precision", "recall", "f1")},
                            **{f"{metric}_signed_error_change": outage_metrics[f"{metric}_signed_estimation_error"] - baseline_metrics[f"{metric}_signed_estimation_error"] for metric in ("precision", "recall", "f1")},
                            **{f"{metric}_absolute_error_change": outage_metrics[f"{metric}_absolute_estimation_error"] - baseline_metrics[f"{metric}_absolute_estimation_error"] for metric in ("precision", "recall", "f1")}
                        })

                    completed_models += 1
                    if completed_models % checkpoint_every == 0:
                        pd.DataFrame(output_rows).to_csv(checkpoint_path, index=False)
                        print(f"Checkpoint saved ({completed_models} model-fold fits, {len(output_rows)} outage conditions)")

    runs = pd.DataFrame(output_rows)
    runs.to_csv(results_dir / "column_outage_runs.csv.gz", index=False)
    checkpoint_path.unlink(missing_ok=True)

    fold_means = runs.groupby([
        "dataset", "model", "outage_fraction_requested", "split_seed", "fold_index"
    ], as_index=False).agg(
        outage_fraction_realized=("outage_fraction_realized", "mean"),
        n_absent_columns=("n_absent_columns", "first"),
        actual_accuracy=("actual_accuracy", "mean"),
        cbpe_accuracy_estimate=("cbpe_accuracy_estimate", "mean"),
        signed_estimation_error=("signed_estimation_error", "mean"),
        absolute_estimation_error=("absolute_estimation_error", "mean"),
        accuracy_interval_coverage=("accuracy_interval_covers_actual", "mean"),
        actual_accuracy_change=("actual_accuracy_change", "mean"),
        signed_error_change=("signed_error_change", "mean"),
        absolute_error_change=("absolute_error_change", "mean")
    )
    summary = fold_means.groupby([
        "dataset", "model", "outage_fraction_requested", "n_absent_columns"
    ], as_index=False).agg(
        n_split_outcomes=("fold_index", "size"),
        outage_fraction_realized=("outage_fraction_realized", "mean"),
        actual_accuracy_mean=("actual_accuracy", "mean"),
        cbpe_accuracy_estimate_mean=("cbpe_accuracy_estimate", "mean"),
        signed_estimation_error_mean=("signed_estimation_error", "mean"),
        absolute_estimation_error_mean=("absolute_estimation_error", "mean"),
        accuracy_interval_coverage=("accuracy_interval_coverage", "mean"),
        actual_accuracy_change_mean=("actual_accuracy_change", "mean"),
        signed_error_change_mean=("signed_error_change", "mean"),
        absolute_error_change_mean=("absolute_error_change", "mean")
    )
    summary.to_csv(results_dir / "column_outage_summary.csv", index=False)
    fold_means.to_csv(results_dir / "column_outage_fold_summary.csv", index=False)
    return runs, summary
