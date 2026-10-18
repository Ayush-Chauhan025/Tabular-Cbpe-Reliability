import json
from pathlib import Path

import pandas as pd


def summarize():
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "default" / "experiment.json"
    results_dir = project_root / "results" / "column_outage_v2_calibration"
    with config_path.open(encoding="utf-8") as stream:
        experiment = json.load(stream)

    rows = []
    for dataset in experiment["datasets"]:
        name = dataset["name"]
        data = pd.read_csv(project_root / "data" / "raw" / f"{name}.csv")
        target_column = dataset["target_column"]
        if dataset.get("drop_duplicates", False):
            data = data.drop_duplicates()

        target = data[target_column]
        if target.dtype == object and set(target.astype(str).str.casefold().unique()) <= {"true", "false"}:
            target = target.astype(str).str.casefold().map({"true": True, "false": False})

        positive_label = dataset["positive_label"]
        counts = target.value_counts()
        negative_labels = [label for label in counts.index if label != positive_label]
        if len(counts) != 2 or len(negative_labels) != 1:
            raise ValueError(f"Expected two classes and positive label {positive_label!r} in {name!r}")

        negative_label = negative_labels[0]
        rows.append({
            "dataset": name,
            "rows": len(data),
            "features": len(data.columns) - 1,
            "target": target_column,
            "positive_class": str(positive_label),
            "positive_count": int(counts[positive_label]),
            "negative_class": str(negative_label),
            "negative_count": int(counts[negative_label])
        })

    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "dataset_overview.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved dataset overview to {output_path}")


if __name__ == "__main__":
    summarize()
