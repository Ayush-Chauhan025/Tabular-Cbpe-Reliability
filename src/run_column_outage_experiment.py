# Run the column-outage experiment

import json
from pathlib import Path

import pandas as pd
import sklearn
import xgboost

from column_outage_experiment import run_column_outage_experiment
from models import MODEL_TYPES, build_model_factory


def main():
    project_root = Path(__file__).resolve().parent.parent
    config_dir = project_root / "configs" / "default"
    with (config_dir / "experiment.json").open(encoding="utf-8") as stream:
        experiment = json.load(stream)
    with (config_dir / "column_outage_experiment.json").open(encoding="utf-8") as stream:
        outage = json.load(stream)

    model_factories = {}
    for model_name in experiment["models"]:
        with (config_dir / f"{model_name}.json").open(encoding="utf-8") as stream:
            parameters = json.load(stream)
        model_factories[model_name] = build_model_factory(MODEL_TYPES[model_name], parameters)

    results_dir = project_root / outage["results_dir"]
    runs, _ = run_column_outage_experiment(
        dataset_configs=experiment["datasets"],
        model_factories=model_factories,
        split_seeds=experiment["split_seeds"],
        n_folds=experiment["n_folds"],
        outage_fractions=outage["outage_fractions"],
        groups_per_split_fraction=outage["groups_per_split_fraction"],
        results_dir=results_dir
    )
    metadata = {
        "source": "column_outage_experiment.py",
        "probability_quality_scores": {
            "brier_score": "mean squared error of positive-class probabilities",
            "ece_10_equal_width": "10-bin equal-width expected calibration error"
        },
        "rows": len(runs),
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__
    }
    (results_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(runs):,} outage conditions to {results_dir}")


if __name__ == "__main__":
    main()
