# Hyperbolic Sheaf Neural Network for Brain Activity Classification

| Network construction | 1-order simplex | 2-order simplex |
| --- | --- | --- |
| Pearson | Pearson + order 1 | Pearson + order 2 |
| Continuity Scaling | CS + order 1 | CS + order 2 |

CS computes cross-scale continuity from ROI time-series signals to detect causal relationships between simplices, whereas PCC uses the correlation matrix.

## Work Environment

Please use a standalone Python environment. Install PyTorch, PyTorch Geometric, and the following dependencies:

```powershell
pip install numpy scipy scikit-learn torch torch-geometric
```

## Data

Set the dataset root:

```powershell
$env:BRAIN_DATA_ROOT = 'your/data/root'
```

The root directory should contain `abide`, `ppmi`, `taowu`, `neurocon`, and `Matai`. Each dataset contains neuroimaging data from individual subjects:

```text
data_root/

└── abide/

    └── subject_control_001/

        ├── *_correlation*.mat

        └── *_features_timeseries*.mat
```

The Pearson experiments load `.mat` matrices whose filenames contain `correlation`. The ACX experiments load ROI × time matrices whose filenames contain `features_timeseries`.

Subject labels are inferred from the subject directory names:

- `control` / `baseline`: control subjects
- `patient` / `postseason`: patient subjects

For PPMI, `prodromal` and `swedd` are also supported.

## Train

```powershell
# Pearson

python -m exp.train --dataset abide --construction_method pearson --simplex_order 1 --cuda 0

python -m exp.train --dataset abide --construction_method pearson --simplex_order 2 --cuda 0

# ACX; multiple CPU processes can be used when computing the ACX cache for the first time

python -m exp.train --dataset abide --construction_method causality --simplex_order 1 --cuda 0 --preprocess_workers 4

python -m exp.train --dataset abide --construction_method causality --simplex_order 2 --cuda 0 --preprocess_workers 4
```

The first run generates preprocessed caches in `dataset/*.pt`. ACX also generates continuity-scaling caches in the corresponding subject directories. Existing caches are reused and will not be recomputed.

Preprocessing can also be performed separately:

```powershell
python -m utils.data_process --dataset abide --construction_method causality --simplex_order 2 --workers 4
```

Training results are saved in `log/run_*`. The `training.log` file records the training process.

## Settings

The fixed hyperparameters for the experiments are specified in [`exp/config.py`](exp/config.py).

## Project Structure

```text
exp/

  train.py              Training entry point and repeated cross-validation

  config.py             Main fixed experimental configuration

  training.py           Single-fold training, validation, and metric computation

models/                 Hyperbolic sheaf diffusion models

utils/

  brain_dataset.py      Subject traversal, preprocessing caches, and dataset objects

  brain_io.py           MAT file and parallel ACX cache I/O

  brain_graph.py        Pearson/ACX graph construction and simplex generation

  continuity_scaling.py ACX continuity-scaling computation

  simplicial.py         Simplicial boundary matrices

lib/                    ODE solvers and hyperbolic geometry operators

dataset/                Automatically generated graph caches

log/                    Training results
```