# Hyperbolic Sheaf Neural Network

Code for Hyperbolic Sheaf Neural Network


## 数据

目录下应包含五个数据集：

```text
abide/
ppmi/
taowu/
neurocon/
Matai/
```

每个受试者目录中使用两类文件：

```text
*_AAL116_correlation_matrix.mat
*_AAL116_features_timeseries.mat
```


## 脑网络构造

### Pearson 构图

Pearson 实验默认读取已有相关矩阵：

```text
*_AAL116_correlation_matrix.mat
```

处理流程：

```text
Pearson matrix -> abs(corr) top_k_percent 筛边 -> PyG graph
```

默认 `--top_k_percent 20`，即保留相关强度绝对值前 20% 的边。

### Causality 构图

causality 实验会自动读取 ROI 时间序列：

```text
*_AAL116_features_timeseries.mat
```

处理流程参考 `bianzhiyu/ContinuityScaling` 中的 `ACX.m`：

```text
ROI time series -> Takens embedding -> ACX-style scaling slope -> connectivity matrix
```

如果受试者目录下已经存在 causality 矩阵，代码会优先直接读取；如果不存在，则从 `*_AAL116_features_timeseries.mat` 重新生成，并保存为：

```text
*_AAL116_causality_matrix.mat
```

默认参数：

```text
--cs_embedding_dim 3
--cs_delay 1
--cs_scales 33
--cs_epsilon_min 0.001
--cs_epsilon_max 1.0
--cs_theiler_window 2
```

## 一阶和二阶结构

一阶脑图：

```text
nodes + edges
```

二阶单纯复形：

```text
nodes + edges + triangles + sigma1 + sigma2
```

二阶结构参考 `Hodge_2Skeleton.m` 的思路：先将连接矩阵二值化为 adjacency，再从闭合三角形构造 2-simplex，并生成 incidence/boundary matrices。

默认 `--max_triangles 500`，用于限制每个样本最多保留 500 个三角形。若要保留全部三角形：

```bash
--max_triangles -1
```

## 训练示例

Pearson 一阶：

```bash
python exp/train.py --dataset abide --construction_method pearson --simplex_order 1
```

Pearson 二阶：

```bash
python exp/train.py --dataset abide --construction_method pearson --simplex_order 2
```

causality 一阶：

```bash
python exp/train.py --dataset abide --construction_method causality --simplex_order 1
```

causality 二阶：

```bash
python exp/train.py --dataset abide --construction_method causality --simplex_order 2
```

五折交叉验证默认开启：

```text
--num_folds 5
```

只运行第一折时：

```bash
python exp/train.py --dataset abide --construction_method pearson --simplex_order 1 --fold_limit 1
```

### 当前默认训练配置

```text
epochs=60
lr=0.003
weight_decay=5e-4
hidden_channels=64
batch_size=32
early_stopping=20
max_t=0.1
step_size=0.1
scheduler=cosine
eval_interval=1
log_interval=1
checkpoint_metric=accuracy
save_model=False
```

`exp/train.sh` 包含 ABIDE Pearson 一阶实验的完整推荐命令。



## 预处理缓存

首次运行会在 `dataset/` 下生成 `.pt` 缓存文件。缓存名包含数据集、构图方式、阶数、top-k、三角形上限和持续缩放参数。

缓存的作用是避免每次训练都重复读取 `.mat`、构图和生成 `sigma1/sigma2`。二阶缓存会保存稀疏 `sigma1` 和预计算的稀疏 Hodge Laplacian；旧版 `sigma1/sigma2` 缓存会在首次读取时自动转换。持续缩放和二阶结构生成较慢，建议保留缓存。

强制重新处理时，删除对应的 `.pt` 和 `_sigma.pt` 文件即可。



## 代码结构

```text
exp/
  parser.py              命令行参数
  train.py               实验入口
  training.py            训练、评估、五折交叉验证
models/
  cont_models.py         Hyperbolic graph/sheaf diffusion 模型
  hyperbolic_ode.py      双曲 ODE / Hodge 修正
  laplacian_builders.py  Sheaf Laplacian 构造
utils/
  brain_dataset.py       数据读取、构图、缓存
  continuity_scaling.py  ACX-style 持续缩放实现
  simplicial.py          2-skeleton、sigma1、sigma2
  data_process.py        独立预处理入口
lib/
  adjoint.py             ODE 求解器
  hyperbolic_geometry.py 双曲几何算子
```

## 依赖说明

训练需要 PyTorch、PyTorch Geometric、`torch_scatter` 和 `torch_sparse`。当前代码可在支持 CUDA 的 Linux/WSL 环境中运行更稳定。

持续缩放预处理依赖 NumPy 和 SciPy。
