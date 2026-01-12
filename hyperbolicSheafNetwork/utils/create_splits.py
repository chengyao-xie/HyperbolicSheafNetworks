import os
import numpy as np
import torch
from torch_geometric.data import Data
from pyG import build_new_brain_graph_dataset, BrainGraphDataset, InMemoryDataset

def create_splits(dataset, train_ratio=0.6, val_ratio=0.2, rng=None):
    num_samples = len(dataset)
    if rng is None:
        rng = np.random
    indices = rng.permutation(num_samples)

    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_mask = np.zeros(num_samples, dtype=np.uint8)
    val_mask = np.zeros(num_samples, dtype=np.uint8)
    test_mask = np.zeros(num_samples, dtype=np.uint8)

    train_mask[train_indices] = 1
    val_mask[val_indices] = 1
    test_mask[test_indices] = 1

    return train_mask, val_mask, test_mask

def save_splits(splits, dataset_name, output_dir, seed=None):
    os.makedirs(output_dir, exist_ok=True)
    train_mask, val_mask, test_mask = splits
    if seed is None:
        filename = f"{dataset_name}_split.npz"
    else:
        # 与 heterophilic.py 中读取文件名保持一致： *_split_0.6_0.2_{seed}.npz
        filename = f"{dataset_name}_split_0.6_0.2_{seed}.npz"

    np.savez(os.path.join(output_dir, filename),
             train_mask=train_mask,
             val_mask=val_mask,
             test_mask=test_mask)

def main():
    this_dir = os.path.dirname(__file__)
    
    dataset_name = "abide"  
    
    # 数据集
    supported_datasets = ['abide', 'Matai', 'neurocon', 'ppmi', 'taowu']
    if dataset_name not in supported_datasets:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {supported_datasets}")
    
    root_dir = os.path.normpath(os.path.join(this_dir, '..', 'data', dataset_name))
    pt_path = os.path.join(this_dir, f"{dataset_name}_graph_labeled_dataset.pt")
    output_dir = os.path.normpath(os.path.join(this_dir, '..', './splits/graph'))

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Raw data directory not found: {root_dir}")

    print(f"Building {dataset_name.upper()} dataset from raw files in: {root_dir}")
    
    data_list = build_new_brain_graph_dataset(root_dir, dataset_name=dataset_name)
        
    if len(data_list) == 0:
        raise RuntimeError("未能从原始文件构建到任何样本，请检查原始数据。")

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), pt_path)
    print(f"Saved processed dataset to {pt_path} (num_samples={len(data_list)})")

    sample_count = len(data_list)
    print(f"Creating graph-level splits for {sample_count} graphs...")
    
    class _LenObj:
        def __init__(self, n): 
            self._n = n
        def __len__(self): 
            return self._n

    dataset_like = _LenObj(sample_count)
    
    # 生成 10 个可复现的划分文件，seed 从 0 到 9
    for seed in range(10):
        rng = np.random.RandomState(seed)
        splits = create_splits(dataset_like, rng=rng)
        save_splits(splits, dataset_name, output_dir, seed=seed)
    
    print(f"Saved 10 graph-level splits for dataset '{dataset_name}' in '{output_dir}' (seeds 0-9).")

if __name__ == "__main__":
    main()