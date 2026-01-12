import os
import torch
import numpy as np
import scipy.io
from torch_geometric.data import Data, InMemoryDataset


def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    for key in mat:
        if not key.startswith('__'):
            return mat[key]
    raise ValueError(f"No valid data found in {file_path}")


# 二阶脑网络
def compute_second_order_matrix(first_order_matrix):
    # Each row is the connectivity profile of a node.
    corr = np.corrcoef(first_order_matrix)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)
    return corr


def brain_graph_dataset(root_dir, top_k_percent=80, graph_type='correlation', order=1):
    data_list = []
    subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    counts = {'control': 0, 'patient': 0, 'prodromal': 0, 'swedd': 0}
    
    for subject in subjects:
        subj_dir = os.path.join(root_dir, subject)
        data_path = None
            
        for f in os.listdir(subj_dir):
            if graph_type.lower() in f.lower():
                data_path = os.path.join(subj_dir, f)
                break

        if data_path is None:
            continue

        data_matrix = load_mat_file(data_path)  
        n_nodes = data_matrix.shape[0]     # ( N , N )
        
        # Only process graphs with 116 nodes (AAL template)
        if n_nodes != 116:
            continue
            
        # Optionally convert to second-order connectivity before building the graph
        if order == 2:
            data_matrix = compute_second_order_matrix(data_matrix)

        x = torch.tensor(data_matrix, dtype=torch.float32)
        
        # 标签
        if 'control' in subject.lower():
        # if 'baseline' in subject.lower():
            graph_label, label_type = 0, 'control'
        elif 'patient' in subject.lower():
        # elif 'postseason' in subject.lower():
            graph_label, label_type = 1, 'patient'
        elif 'prodromal' in subject.lower():
            graph_label, label_type = 2, 'prodromal'
        elif 'swedd' in subject.lower():
            graph_label, label_type = 3, 'swedd'
        else:
            continue

        counts[label_type] += 1
        
        # 构建边，选取权重前80%的边
        connectivities = []
        positions = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                abs_conn = abs(data_matrix[i, j])
                connectivities.append(abs_conn)
                positions.append((i, j, data_matrix[i, j]))
        
        threshold = np.percentile(connectivities, 100 - top_k_percent)
        
        edge_index = [[i, j] for i, j, conn_val in positions if abs(conn_val) >= threshold 
                  for i, j in [[i, j], [j, i]]]
        edge_weights = [conn_val for i, j, conn_val in positions if abs(conn_val) >= threshold 
                for _ in [0, 1]]
        
        if not edge_index:
            continue

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    y=torch.tensor(graph_label, dtype=torch.long))
        data.subject_id = subject
        data.n_nodes = n_nodes
        data_list.append(data)

    print(f"Successfully processed {len(data_list)} subjects:")
    print(f" Control: {counts['control']}, Patient: {counts['patient']}, "
          f"Prodromal: {counts['prodromal']}, SWEDD: {counts['swedd']}")
    
    return data_list


class BrainGraphDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, graph_type='correlation', top_k_percent=80, order=1, transform=None, pre_transform=None):
        self.top_k_percent = top_k_percent
        self.dataset_name = dataset_name
        self.graph_type = graph_type
        self.order = order
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.dataset_name}_{self.graph_type}_order{self.order}_brain_graph_data_top{self.top_k_percent}.pt']

    def download(self):
        pass

    def process(self):
        data_list = brain_graph_dataset(self.root, self.top_k_percent, self.graph_type, order=self.order)
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def num_classes(self):
        return 4 if self.dataset_name.lower() == "ppmi" else 2
        
    @property
    def num_features(self):
        return self.data.x.size(1)

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)}, order={self.order}, top_k_percent={self.top_k_percent})'


def get_dataset(name, graph_type='correlation', order=1):
    dataset_paths = {
        'abide': "/home/featurize/data/abide",
        'ppmi': "/home/featurize/data/ppmi",
        'taowu': "/home/featurize/data/taowu",
        'neurocon': "/home/featurize/data/neurocon",
        'Matai': "/home/featurize/data/Matai",
    }
    
    root_dir = dataset_paths[name]
    dataset = BrainGraphDataset(root=root_dir, dataset_name=name, graph_type=graph_type, top_k_percent=80, order=order)
    dataset.name = name
    return dataset


def create_stratified_splits(labels, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, rng=None):
    if rng is None:
        rng = np.random
    
    num_samples = len(labels)
    unique_classes = np.unique(labels)
    
    train_mask = np.zeros(num_samples, dtype=np.uint8)
    val_mask = np.zeros(num_samples, dtype=np.uint8)
    test_mask = np.zeros(num_samples, dtype=np.uint8)
    
    # 对每个类别分别进行分层采样
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        cls_indices_shuffled = rng.permutation(cls_indices)
        
        n_cls = len(cls_indices)
        n_train = int(n_cls * train_ratio)
        n_val = int(n_cls * val_ratio)
        
        train_indices = cls_indices_shuffled[:n_train]
        val_indices = cls_indices_shuffled[n_train:n_train + n_val]
        test_indices = cls_indices_shuffled[n_train + n_val:]
        
        train_mask[train_indices] = 1
        val_mask[val_indices] = 1
        test_mask[test_indices] = 1
    
    return train_mask, val_mask, test_mask