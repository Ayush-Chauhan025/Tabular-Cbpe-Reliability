# Column-outage experiment protocol

## Question

Can confidence-based performance monitoring estimate a classifier's accuracy when one or more required input columns are absent from an incoming batch?

## Relationship to missing-feature research

Recent work studies prediction or adaptation when tabular features are missing, including feature-decrement test-time adaptation and models trained to avoid relying on missing values. This experiment addresses a different outcome: whether a frozen model's confidence-based accuracy estimate remains close to its measured accuracy under column outages. The estimator follows [Confidence-based Performance Estimation (CBPE)](https://link.springer.com/article/10.1007/s10994-025-06970-3); the contribution is an empirical reliability evaluation, not a new estimator, imputation method, or missing-feature prediction method. Related feature-outage papers include [Fully Test-Time Adaptation for Feature Decrement in Tabular Data](https://www.ijcai.org/proceedings/2025/550) and [Prediction Models that Learn to Avoid Missing Values](https://proceedings.mlr.press/v267/stempfle25a.html).

## Operational scenario

The fitted classifier and preprocessing pipeline were trained with the complete schema. At evaluation, selected test columns are removed from the incoming data frame, as they would be when the upstream source does not provide them. A schema adapter restores the expected column order and inserts missing values for absent columns; the training-fitted median or most-frequent imputer then supplies the model-compatible values. This tests an actual absent-column request followed by a predeclared recovery step. Without schema alignment, the fitted transformer cannot accept the incomplete schema.

The adapter output is checked against a control in which the same columns are present but all values are missing. They must be exactly equal before preprocessing. This confirms that the column-omission protocol preserves the established missing-value semantics and isolates the change in the incoming schema.

## Outage severity and evaluation

The benchmark covers Credit, Heart Disease, MiniBooNE, and Electricity with Logistic Regression, Random Forest, linear SVM, and XGBoost. Credit, Heart Disease, and MiniBooNE use 10 split seeds and stratified five-fold cross-fitting (60% train, 20% calibration, 20% test per fold); Electricity uses four forward-rolling windows from one series. Each split fits preprocessing and the classifier on its training portion, then fits sigmoid probability calibration on a separate clean calibration portion. Numeric features are median-imputed and standardized, and categorical features are mode-imputed and one-hot encoded. All preprocessing is fitted on training data only. The fitted preprocessing, classifier, and calibrator are frozen for the outage test.

For each dataset, model, and existing train/calibration/test split, up to 20 unique feature groups are selected at each requested outage fraction (10%, 25%, and 50%). If fewer than 20 unique groups are possible, every group is evaluated. The random generator is seeded from the split seed and fold index; feature-group selection does not use labels or outcomes. The fractions are rounded up to whole columns. All features in a selected group are absent for every row in that test batch. The fitted model and clean-data calibrator remain fixed. CBPE accuracy estimates, actual accuracy, signed and absolute estimation error, and nominal 95% interval coverage are recorded.

The fixed model and calibrator match the monitoring setting in which an already-deployed model experiences an outage. Summary statistics first average sampled feature groups within each model/split and then average across split outcomes. For random splits, supplementary paired bootstrap ranges resample the 10 split seeds as clusters after averaging folds and models within each seed. Group-level quantiles are descriptive because their conditions overlap by model and split. Electricity is one time series evaluated across four forward windows; its bootstrap ranges are descriptive only.

## Interpretation

This experiment studies frozen-model monitoring with a schema adapter and training-fitted imputation. It does not claim that an unadapted serving stack can process a missing schema, that retraining is unnecessary in all applications, or that the imputed model is optimal. Results are conditional on the specified outage fractions, replacement mechanism, classifiers, datasets, split design, and calibration procedure.

## Generated outputs

The raw conditions, split-averaged tables, mask-variation summaries, seed-bootstrap summaries, figures, and summary reports are written under `results/column_outage_v2_calibration/`. The raw condition table is gzip-compressed as `column_outage_runs.csv.gz`. Run the commands in the repository README to reproduce the experiment and summaries.
