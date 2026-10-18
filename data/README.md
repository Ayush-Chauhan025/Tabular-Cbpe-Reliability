# Dataset files

The experiment reads four prepared CSV files from `data/raw/`. They are not included in this repository; obtain them from their providers and follow each provider's terms. For exact compatibility, use the prepared files distributed with TabFSBench rather than independently transforming the upstream records.

| Expected file | Prepared source | Upstream dataset |
|---|---|---|
| `credit.csv` | [TabFSBench Credit file](https://github.com/LAMDA-NeSy/TabFSBench/blob/main/dataset/credit/credit.csv) | [OpenML credit-g, dataset 31](https://www.openml.org/search?id=31&sort=runs&type=data) |
| `heart.csv` | [TabFSBench Heart file](https://github.com/LAMDA-NeSy/TabFSBench/blob/main/dataset/heart/heart.csv) | [Heart Failure Prediction, Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| `miniboone.csv` | [TabFSBench MiniBooNE file](https://github.com/LAMDA-NeSy/TabFSBench/blob/main/dataset/miniboone/miniboone.csv) | [MiniBooNEn, Kaggle](https://www.kaggle.com/datasets/alexanderliapatis/miniboone) |
| `electricity.csv` | [TabFSBench Electricity file](https://github.com/LAMDA-NeSy/TabFSBench/blob/main/dataset/electricity/electricity.csv) | [Electricity, Kaggle](https://www.kaggle.com/datasets/vstacknocopyright/electricity) |

The following SHA-256 values identify the prepared CSVs used for the checked-in run. Compute a checksum after download with `Get-FileHash .\data\raw\<file> -Algorithm SHA256` in PowerShell.

| File | SHA-256 |
|---|---|
| `credit.csv` | `CB138FA575316D50AD7DF9ABF681EB6DADF175398281FB5D7CC51DB7208A4014` |
| `heart.csv` | `9A88C8DCAEDC1302C69AE11529F068E1A1F49ACC3A48D81841CB504FFF872828` |
| `miniboone.csv` | `6F8B75A16BF8364169A0CF3BA78D56C8A44CDE376025864F1648BC7176B4E18A` |
| `electricity.csv` | `3396057A219B0591DBACD93B3EB17A056F89C3626456F843834B9C4B50306E04` |

The MiniBooNE configuration removes exact duplicate rows before splitting; the checksum above is for the prepared input CSV before that step.
