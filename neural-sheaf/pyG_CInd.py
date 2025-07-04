import os
import torch
import numpy as np
import scipy.io
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp


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


def build_brain_graph_dataset_from_cind(root_dir):
    print('=== 构建基于 CInd 的脑图数据集（无 threshold，全连接非对角） ===')
    data_list = []
    subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for subject in subjects:
        subj_dir = os.path.join(root_dir, subject)

        causality_path = find_file_with_keywords(subj_dir, ["causality"])
        ts_path = find_file_with_keywords(subj_dir, ["timeseries", "AAL"])

        if causality_path is None or ts_path is None:
            print(f"Skipping {subject} due to missing causality or timeseries file.")
            continue

        mat = scipy.io.loadmat(causality_path)
        if 'CInd' not in mat:
            print(f"Skipping {subject}, no CInd in {causality_path}")
            continue
        corr = mat['CInd']  # [116 x 116]

        ts = load_mat_file(ts_path)      # [T x 116]
        if corr.shape != (116, 116) or ts.shape[1] != 116:
            print(f"Shape mismatch in {subject}")
            continue

        node_features = np.mean(ts, axis=0)  # [116,]
        node_labels = (node_features > 0).astype(np.int64)  # [116]

        x = torch.tensor(node_features, dtype=torch.float32).view(-1, 1)  # [116,1]
        y = torch.tensor(node_labels, dtype=torch.long)                   # [116]

        edge_index = []
        for i in range(116):
            for j in range(116):
                if i != j:
                    edge_index.append([i, j])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.subject_id = subject
        data_list.append(data)

    return data_list
class BrainGraphCINDDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['brain_graph_data_cind.pt']

    def download(self):
        pass

    def process(self):
        data_list = build_brain_graph_dataset_from_cind(self.root)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)})'
if __name__ == "__main__":
    root_dir = "./dataset/neurocon"
    dataset = BrainGraphCINDDataset(root=root_dir)
    print(f"Loaded {len(dataset)} subjects from CInd graphs.")
    torch.save(dataset, "brain_node_labeled_dataset_cind.pt")
