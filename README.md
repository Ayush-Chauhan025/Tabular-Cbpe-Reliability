# CBPE when test-time columns are missing

This repository contains the code and saved results for a study of whether confidence-based performance estimation (CBPE) tracks a classifier's measured performance when input columns are unavailable at test time.

The experiment keeps each trained classifier and probability calibrator fixed. Before prediction, the code restores the missing columns in the expected order and fills them with values from preprocessing fitted on the training data. It compares CBPE estimates with accuracy, precision, recall, and F1 calculated from held-out labels. The evaluation uses Credit, Heart Disease, MiniBooNE, and Electricity, with Logistic Regression, Random Forest, linear SVM, and XGBoost.

The findings apply to this model, schema-restoration, and imputation setup. They do not show that every confidence-based monitor fails when features are missing, and the study does not compare CBPE with other monitors.

## What's in the repository

- `src/` contains the experiment, data preparation, model, estimation, and analysis code.
- `configs/default/` contains the dataset, split, model, and outage settings.
- `data/README.md` lists the prepared dataset sources, filenames, and checksums. The CSV files are not included.
- `docs/column_outage_protocol.md` describes the experiment and its limits.
- `results/column_outage_v2_calibration/` contains the recorded run data and some summaries. The full condition table is compressed.

## Prepare the environment and data

The saved run used Python 3.13, scikit-learn 1.9.0, XGBoost 3.4.1, and pandas 3.0.5. Install the listed packages from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Download the prepared CSVs described in `data/README.md` and put them in `data/raw/` using these filenames:

```text
data/raw/credit.csv
data/raw/heart.csv
data/raw/miniboone.csv
data/raw/electricity.csv
```

The experiment expects the prepared files distributed with TabFSBench. It does not apply an independent preprocessing recipe to the upstream datasets. The loader checks that each file and target column is present.

## Recreate the results

To fit the models and run the full experiment, use:

```powershell
python src/run_column_outage_experiment.py
```

This run uses 10 seeds and stratified five-fold cross-fitting for Credit, Heart Disease, and MiniBooNE. Electricity is evaluated in four chronological windows from one time series. The full run can take a while.

The repository includes the recorded per-condition run data, so you can regenerate the analyses without fitting the models again:

```powershell
python src/summarize_dataset_overview.py
python src/summarize_column_outage_metrics.py
python src/summarize_calibration_quality.py
python src/build_column_outage_workflow.py
```

These scripts write the dataset counts, metric summaries and plots, probability-quality results, and workflow figure under `results/column_outage_v2_calibration/`.

## License

The code in this repository is available under the MIT License; see `LICENSE`. The license applies to the repository's code and documentation, not to the third-party datasets. Each dataset has its own terms; check those with the dataset provider.
