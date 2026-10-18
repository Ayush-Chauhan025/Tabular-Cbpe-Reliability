import numpy as np
from scipy.stats import norm


def _normal_interval(estimate, variance, confidence_level):
    alpha = 1.0 - confidence_level
    margin = norm.ppf(1.0 - alpha / 2.0) * np.sqrt(max(float(variance), 0.0))
    return {
        "estimate": float(estimate),
        "ci_lower": float(np.clip(estimate - margin, 0.0, 1.0)),
        "ci_upper": float(np.clip(estimate + margin, 0.0, 1.0))
    }


def _ratio_delta_interval(numerator_mean, denominator_mean, numerator_variance,
    denominator_variance, denominator_constant, confidence_level, scale_numerator=1.0):
    denominator = numerator_mean + denominator_mean + denominator_constant
    if denominator <= 0:
        return {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    estimate = scale_numerator * numerator_mean / denominator
    derivative_numerator = scale_numerator * (denominator - numerator_mean) / (denominator ** 2)
    derivative_other = -scale_numerator * numerator_mean / (denominator ** 2)
    variance = (derivative_numerator ** 2) * numerator_variance + (derivative_other ** 2) * denominator_variance
    return _normal_interval(estimate, variance, confidence_level)


def estimate_cbpe_metrics(probabilities, predictions, classes, positive_label="bad", confidence_level=0.95):
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions)
    classes = np.asarray(classes)

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("probabilities must have shape (n_samples, 2)")
    if len(probabilities) == 0 or len(predictions) != len(probabilities):
        raise ValueError("probabilities and predictions must have the same nonzero length")
    if len(classes) != 2 or positive_label not in classes:
        raise ValueError("classes must contain exactly two labels, including positive_label")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("class probabilities must be finite and in [0, 1]")

    positive_index = int(np.where(classes == positive_label)[0][0])
    positive_scores = probabilities[:, positive_index]
    predicted_positive = predictions == positive_label
    n_positive_predictions = int(predicted_positive.sum())
    n_negative_predictions = len(predictions) - n_positive_predictions

    correct_probabilities = np.where(predicted_positive, positive_scores, 1.0 - positive_scores)
    accuracy_estimate = float(correct_probabilities.mean())
    accuracy_variance = float(np.sum(correct_probabilities * (1.0 - correct_probabilities)) / len(predictions) ** 2)
    accuracy = _normal_interval(accuracy_estimate, accuracy_variance, confidence_level)

    tp_probabilities = positive_scores[predicted_positive]
    fn_probabilities = positive_scores[~predicted_positive]
    expected_tp = float(tp_probabilities.sum())
    expected_fn = float(fn_probabilities.sum())
    tp_variance = float(np.sum(tp_probabilities * (1.0 - tp_probabilities)))
    fn_variance = float(np.sum(fn_probabilities * (1.0 - fn_probabilities)))

    if n_positive_predictions:
        precision = _normal_interval(expected_tp / n_positive_predictions,
            tp_variance / n_positive_predictions ** 2, confidence_level)
    else:
        precision = {"estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    recall = _ratio_delta_interval(expected_tp, expected_fn, tp_variance, fn_variance,
        denominator_constant=0.0, confidence_level=confidence_level)
    f1 = _ratio_delta_interval(expected_tp, expected_fn, tp_variance, fn_variance,
        denominator_constant=float(n_positive_predictions), confidence_level=confidence_level,
        scale_numerator=2.0)

    result = {
        "expected_tp": expected_tp,
        "expected_fp": float(n_positive_predictions - expected_tp),
        "expected_fn": expected_fn,
        "expected_tn": float(n_negative_predictions - expected_fn),
        "interval_method": "normal_delta_approximation"
    }
    for metric_name, metric_summary in (("accuracy", accuracy), ("precision", precision), ("recall", recall), ("f1", f1)):
        result[f"{metric_name}_estimate"] = metric_summary["estimate"]
        result[f"{metric_name}_ci_lower"] = metric_summary["ci_lower"]
        result[f"{metric_name}_ci_upper"] = metric_summary["ci_upper"]

    for metric_name in ("precision", "recall", "f1"):
        result[f"{metric_name}_bad_estimate"] = result[f"{metric_name}_estimate"]
        result[f"{metric_name}_bad_ci_lower"] = result[f"{metric_name}_ci_lower"]
        result[f"{metric_name}_bad_ci_upper"] = result[f"{metric_name}_ci_upper"]

    return result
