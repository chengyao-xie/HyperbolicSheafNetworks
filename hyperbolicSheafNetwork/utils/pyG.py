import os
import torch
import numpy as np
import scipy.io
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp
import importlib


def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    for key in mat:
        if not key.startswith('__'):
            return mat[key]
    raise ValueError(f"No valid data found in {file_path}")


def find_file_with_keywords(directory, required_keywords):
    for f in os.listdir(directory):
        if f.endswith(".mat") and all(keyword in f for keyword in required_keywords):
            return os.path.join(directory, f)
    return None


def build_new_brain_graph_dataset(root_dir, dataset_name, threshold=0.1):
    """
    构建脑图数据，包含正确的诊断标签
    """
    print(f"Building {dataset_name.upper()} brain graph dataset from: {root_dir}")
    data_list = []
    subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    control_count = 0
    patient_count = 0
    prodromal_count = 0
    swedd_count = 0
    
    for subject in subjects:
        subj_dir = os.path.join(root_dir, subject)

        corr_path = None
        ts_path = None
        
        try:
            files = os.listdir(subj_dir)
        except OSError:
            print(f"Cannot access directory {subj_dir}")
            continue
            
        for f in files:
            if not f.endswith('.mat'):
                continue
                
            f_lower = f.lower()
            
            # 相关性矩阵文件
            if 'correlation' in f_lower and 'aal' in f_lower:
            # if 'causality' in f_lower
                corr_path = os.path.join(subj_dir, f)
                
            # 时间序列文件
            if 'timeseries' in f_lower and 'aal' in f_lower:
                ts_path = os.path.join(subj_dir, f)

        if corr_path is None or ts_path is None:
            # 未找到相关文件
            print(f"Skipping {subject} due to missing required files:")
            print(f"  Directory: {subj_dir}")
            print(f"  Files found: {files}")
            print(f"  corr_path: {corr_path}")
            print(f"  ts_path: {ts_path}")
            continue

        try:
            corr = load_mat_file(corr_path)  # 连接性矩阵
            ts = load_mat_file(ts_path)      # 时间序列数据
        except Exception as e:
            print(f"Error loading files for {subject}: {e}")
            continue

        # 检查数据维度
        if len(corr.shape) != 2 or corr.shape[0] != corr.shape[1]:
            print(f"Invalid correlation matrix shape for {subject}: {corr.shape}")
            continue
            
        if len(ts.shape) != 2:
            print(f"Invalid time series shape for {subject}: {ts.shape}")
            continue
            
        # 确保时间序列和连接性矩阵的节点数一致
        n_nodes = corr.shape[0]
        if ts.shape[1] != n_nodes:
            print(f"Dimension mismatch for {subject}: corr {corr.shape}, ts {ts.shape}")
            continue

        # 节点特征使用所有时间点，转置为 [n_nodes, T]
        node_features = ts.T  # [n_nodes, T]
        
        x = torch.tensor(node_features, dtype=torch.float32)  # [n_nodes, T]
        
        # 生成图标签
        if 'control' in subject.lower():
        # if 'baseline' in subject.lower():
            graph_label = 0  # 正常人
            control_count += 1
        elif 'patient' in subject.lower():
        #elif 'postseason' in subject.lower():
            graph_label = 1  # 患者
            patient_count += 1
        
        elif 'prodromal' in subject.lower():
            graph_label = 2
            prodromal_count += 1
        elif 'swedd' in subject.lower():
            graph_label = 3
            swedd_count += 1
        else:
            # 如果无法从文件夹名判断，则跳过该被试
            print(f"Warning: Cannot determine label for subject {subject}, skipping.")
            continue

        # 构建边（基于连接性矩阵的阈值）
        edge_index = []
        edge_weights = []
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if abs(corr[i, j]) > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_weights.extend([corr[i, j], corr[i, j]])
        
        if len(edge_index) == 0:
            print(f"No edges above threshold {threshold} for {subject}, using fully connected graph.")
            # 如果没有边超过阈值，使用全连接图
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i != j:
                        edge_index.append([i, j])
                        edge_weights.append(corr[i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)

        # 创建数据对象（图分类任务）
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.subject_id = subject
        data.graph_label = torch.tensor(graph_label, dtype=torch.long)
        data.n_nodes = n_nodes
        
        data_list.append(data)
        # print(f"Successfully processed {subject}: {n_nodes} nodes, {len(edge_index[0])} edges, label={graph_label}")

    print(f"Successfully processed {len(data_list)} subjects:")
    print(f" Control subjects: {control_count}")
    print(f" Patient subjects: {patient_count}")
    print(f" Prodromal subjects: {prodromal_count}")
    print(f" SWEDD subjects: {swedd_count}")
    
    return data_list


class BrainGraphDataset(InMemoryDataset):
    """处理脑图数据的图分类任务"""
    
    def __init__(self, root, dataset_name, threshold=0.1, transform=None, pre_transform=None):
        self.threshold = threshold
        self.dataset_name = dataset_name
        super().__init__(root, transform, pre_transform)

        # 尝试安全加载
        try:
            module = importlib.import_module("torch_geometric.data.data")
            DataEdgeAttr = getattr(module, "DataEdgeAttr", None)
            if DataEdgeAttr is not None:
                torch.serialization.add_safe_globals([DataEdgeAttr])
            else:
                torch.serialization.add_safe_globals(["torch_geometric.data.data.DataEdgeAttr"])
        except Exception:
            try:
                torch.serialization.add_safe_globals(["torch_geometric.data.data.DataEdgeAttr"])
            except Exception:
                pass

        # 加载处理好的数据
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
        except Exception:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}_brain_graph_data_th{self.threshold}.pt']

    def download(self):
        pass

    def process(self):
        data_list = build_new_brain_graph_dataset(self.root, self.dataset_name, self.threshold)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 保存处理后的数据
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_classes(self):
        if self.dataset_name.lower() == "ppmi":
            return 4
        else:
            return 2
        
    @property
    def num_features(self):
        """返回节点特征数"""
        return self.data.x.size(1) if hasattr(self.data, 'x') and self.data.x is not None else 1

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)}, threshold={self.threshold})'