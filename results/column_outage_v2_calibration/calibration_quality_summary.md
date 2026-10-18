# Probability quality under test-time column outage

Brier score is the mean squared error between positive-class probabilities and the observed labels. ECE uses 10 equal-width bins to summarize calibration; its value depends on the binning scheme. Test labels are used only to calculate these offline scores and are never passed to CBPE.

The table averages outage groups within each model and split, then weights model/split outcomes equally. The baseline is paired by the same model/split. Requested outage fractions are rounded up to whole columns; report the number removed and realized share when comparing datasets.

| Dataset | Requested outage | Columns removed | Brier score | Change vs baseline | 10-bin ECE | Change vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| Credit | 0% | 0 | 0.1685 | 0.0000 | 0.0639 | 0.0000 |
| Credit | 10% | 2 | 0.1750 | +0.0066 | 0.0728 | +0.0090 |
| Credit | 25% | 5 | 0.1851 | +0.0166 | 0.0874 | +0.0235 |
| Credit | 50% | 10 | 0.2016 | +0.0331 | 0.1097 | +0.0458 |
| Heart Disease | 0% | 0 | 0.1038 | 0.0000 | 0.0636 | 0.0000 |
| Heart Disease | 10% | 2 | 0.1310 | +0.0272 | 0.0965 | +0.0329 |
| Heart Disease | 25% | 3 | 0.1464 | +0.0426 | 0.1132 | +0.0496 |
| Heart Disease | 50% | 6 | 0.1953 | +0.0915 | 0.1610 | +0.0974 |
| MiniBooNE | 0% | 0 | 0.0730 | 0.0000 | 0.0303 | 0.0000 |
| MiniBooNE | 10% | 5 | 0.0879 | +0.0149 | 0.0472 | +0.0169 |
| MiniBooNE | 25% | 13 | 0.1202 | +0.0471 | 0.0845 | +0.0542 |
| MiniBooNE | 50% | 25 | 0.1892 | +0.1161 | 0.1616 | +0.1314 |
| Electricity | 0% | 0 | 0.2100 | 0.0000 | 0.1424 | 0.0000 |
| Electricity | 10% | 1 | 0.2179 | +0.0079 | 0.1540 | +0.0116 |
| Electricity | 25% | 2 | 0.2226 | +0.0126 | 0.1570 | +0.0146 |
| Electricity | 50% | 4 | 0.2385 | +0.0285 | 0.1620 | +0.0196 |

The Brier score reflects both reliability and resolution, so it is not a pure calibration-error measure. ECE is sensitive to the selected binning scheme and sample size. These results describe the tested held-out conditions; they do not establish that calibration change caused CBPE estimation error.
