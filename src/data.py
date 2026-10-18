from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def get_tabular_splits(dataset_config, random_state=42, n_folds=5):
    dataset_name = dataset_config["name"]
    target_column = dataset_config["target_column"]
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / f"{dataset_name}.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data = pd.read_csv(data_path)
    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} not found in {dataset_name!r}")
    target_values = data[target_column].dropna()
    if target_values.dtype == object and set(target_values.astype(str).str.casefold().unique()) <= {"true", "false"}:
        data[target_column] = data[target_column].astype(str).str.casefold().map({"true": True, "false": False})
    if dataset_config.get("drop_duplicates", False):
        before = len(data)
        data = data.drop_duplicates().reset_index(drop=True)
        print(f"Removed {before - len(data)} exact duplicate rows from {dataset_name}")
    if data[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values")
    if data[target_column].nunique() != 2:
        raise ValueError(f"Target column {target_column!r} must contain exactly two classes")

    if dataset_config.get("split_strategy", "stratified_crossfit") == "chronological":
        order_columns = dataset_config.get("order_columns", [])
        missing_columns = [column for column in order_columns if column not in data.columns]
        if not order_columns or missing_columns:
            raise ValueError(f"Invalid chronological order columns: {missing_columns or order_columns}")
        data = data.sort_values(order_columns, kind="mergesort").reset_index(drop=True)
        windows = dataset_config.get("chronological_windows", [
            {"train_end": 0.60, "calibration_end": 0.80, "test_end": 1.0}
        ])
        partitions = []
        previous_test_end = 0.0
        previous_train_end = 0.0
        for window in windows:
            train_end = window["train_end"]
            calibration_end = window["calibration_end"]
            test_end = window["test_end"]
            if not 0 < train_end < calibration_end < test_end <= 1:
                raise ValueError(f"Invalid chronological window: {window}")
            if train_end <= previous_train_end or calibration_end < previous_test_end:
                raise ValueError("Chronological windows must expand training data and use non-overlapping test periods")
            train_boundary = int(len(data) * train_end)
            calibration_boundary = int(len(data) * calibration_end)
            test_boundary = int(len(data) * test_end)
            train_start = 0
            calibration_start = train_boundary
            test_start = calibration_boundary
            partitions.append((data.iloc[train_start:train_boundary],
                data.iloc[calibration_start:calibration_boundary], data.iloc[test_start:test_boundary]))
            previous_test_end = test_end
            previous_train_end = train_end
        strategy = f"chronological_rolling_{len(partitions)}_windows"
    else:
        if n_folds < 3:
            raise ValueError("n_folds must be at least 3")
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        folds = [indices for _, indices in splitter.split(data, data[target_column])]
        partitions = []
        for fold_index in range(n_folds):
            test_indices = folds[fold_index]
            calibration_indices = folds[(fold_index + 1) % n_folds]
            excluded = {fold_index, (fold_index + 1) % n_folds}
            train_indices = [index for other_fold, values in enumerate(folds) if other_fold not in excluded for index in values]
            partitions.append((data.iloc[train_indices], data.iloc[calibration_indices], data.iloc[test_indices]))
        strategy = f"stratified_{n_folds}_fold_crossfit"

    splits = []
    for fold_index, (train_data, calibration_data, test_data) in enumerate(partitions):
        x_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]
        x_cal = calibration_data.drop(columns=[target_column])
        y_cal = calibration_data[target_column]
        x_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        if y_train.nunique() != 2 or y_cal.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(f"Split {fold_index} for {dataset_name} does not contain both target classes")

        print(f"{dataset_name} split {fold_index + 1}/{len(partitions)} ({strategy})")
        for name, labels in (("train", y_train), ("calibration", y_cal), ("test", y_test)):
            print(name, len(labels), labels.value_counts(normalize=True).round(3).to_dict())
        splits.append((x_train, y_train, x_test, y_test, x_cal, y_cal))

    return splits
