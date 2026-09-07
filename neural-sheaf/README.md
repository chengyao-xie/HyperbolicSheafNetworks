# Hyperbolic Sheaf Neural Network for Brain Activity Classification



| 图构造 | 一阶单纯形 | 二阶单纯形 |
| --- | --- | --- |
| Pearson | Pearson + order 1 | Pearson + order 2 |
| ACX 持续缩放 | ACX + order 1 | ACX + order 2 |

ACX 以 ROI 时间序列计算有向连续缩放强度，并保留小、中、大三个尺度的强度和可靠性；Pearson 直接使用相关矩阵。二阶模型额外使用边—点和面—边边界矩阵。

## Work environment

建议使用独立的 Python 环境，并安装 PyTorch、PyTorch Geometric 及以下依赖：
```powershell
pip install numpy scipy scikit-learn torch torch-geometric
```


## Data

数据根目录：
```powershell
$env:BRAIN_DATA_ROOT = 'your/data/root'
```

目录下可包含 `abide`、`ppmi`、`taowu`、`neurocon` 和 `Matai`。每个数据集由受试者目录组成：

```text
data_root/
└── abide/
    └── subject_control_001/
        ├── *_correlation*.mat
        └── *_features_timeseries*.mat
```

Pearson 实验读取文件名包含 `correlation` 的 `.mat` 矩阵；ACX 实验读取文件名包含 `features_timeseries` 的 ROI × time 矩阵。类别由受试者目录名识别：`control`/`baseline` 为对照，`patient`/`postseason` 为患者；PPMI 还支持 `prodromal` 与 `swedd`。

## Train
```powershell
# Pearson
python -m exp.train --dataset abide --construction_method pearson --simplex_order 1 --cuda 0
python -m exp.train --dataset abide --construction_method pearson --simplex_order 2 --cuda 0

# ACX；首次计算 ACX 缓存时可用多个 CPU 进程
python -m exp.train --dataset abide --construction_method causality --simplex_order 1 --cuda 0 --preprocess_workers 4
python -m exp.train --dataset abide --construction_method causality --simplex_order 2 --cuda 0 --preprocess_workers 4
```

首次运行会生成 `dataset/*.pt` 预处理缓存。ACX 还会在各受试者目录生成持续缩放缓存；已有缓存不会重复计算。也可以先单独预处理：

```powershell
python -m utils.data_process --dataset abide --construction_method causality --simplex_order 2 --workers 4
```

训练结果保存在 `log/run_*`：`training.log` 记录训练过程.

## Setting
实验的固定超参数位于 [exp/config.py](exp/config.py)。

## 项目结构

```text
exp/
  train.py              训练入口与重复交叉验证
  config.py             主实验固定配置
  training.py           单折训练、验证与指标计算
models/                 双曲 sheaf 扩散模型
utils/
  brain_dataset.py      样本遍历、预处理缓存与数据集对象
  brain_io.py           MAT 文件与并行 ACX 缓存读写
  brain_graph.py        Pearson/ACX 构图与单纯形构造
  continuity_scaling.py ACX 持续缩放计算
  simplicial.py         单纯形边界矩阵
lib/                    ODE 求解与双曲几何算子
dataset/                自动生成的图缓存
log/                    训练结果
```
