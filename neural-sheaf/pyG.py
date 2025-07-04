
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


def build_brain_graph_dataset(root_dir, threshold=0.3):
    print('111111111111111111111111111111111111111111111111111111111111111111111')
    data_list = []
    subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    for subject in subjects:
        subj_dir = os.path.join(root_dir, subject)

        corr_path = find_file_with_keywords(subj_dir, ["correlation", "AAL"])
        ts_path = find_file_with_keywords(subj_dir, ["timeseries", "AAL"])

        if corr_path is None or ts_path is None:
            print(f"Skipping {subject} due to missing AAL files.")
            continue

        corr = load_mat_file(corr_path)  # [116 x 116]
        ts = load_mat_file(ts_path)      # [T x 116]

        if corr.shape != (116, 116) or ts.shape[1] != 116:
            print(f"Shape mismatch in {subject}")
            continue

        # === 改动点：每个节点一个标签，这里示例用节点时间序列均值的阈值分割 ===
        node_features = np.mean(ts, axis=0)  # [116,]
        # 简单阈值分类，均值 > 0 为1， <=0 为0
        node_labels = (node_features > 0).astype(np.int64)  # [116,] 每个节点标签

        x = torch.tensor(node_features, dtype=torch.float32).view(-1, 1)  # [116,1]
        y = torch.tensor(node_labels, dtype=torch.long)                   # [116]
        print(y.shape)

        edge_index = []
        for i in range(116):
            for j in range(i + 1, 116):
                if abs(corr[i, j]) > threshold:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
        if len(edge_index) == 0:
            print(f"No edges for {subject}, skipping.")
            continue

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index, y=y)
        data.subject_id = subject
        data_list.append(data)

    return data_list


from torch_geometric.data import InMemoryDataset

class BrainGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        # 加载缓存文件，父类__init__会自动调用process()（如果缓存不存在）
        self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        # 你可以根据需要调整 raw 文件夹路径
        return osp.join(self.root, 'raw')

    @property
    def processed_dir(self):
        # 处理后缓存文件路径
        return osp.join(self.root, 'processed')

    @property
    def raw_file_names(self):
        # 如果没有原始文件，就返回空列表，表示不需要download
        return []

    @property
    def processed_file_names(self):
        # 缓存文件名
        return ['brain_graph_data.pt']

    def download(self):
        # 不需要下载数据，直接pass
        pass

    def process(self):
        # 生成数据列表
        data_list = build_brain_graph_dataset(self.root)

        # 应用预处理
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 合并成批，保存为缓存
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.__class__.__name__}({len(self)})'


if __name__ == "__main__":
    root_dir = "D:/脑网络/neurocon/neurocon"
    dataset = BrainGraphDataset(root=root_dir)
    print(f"Loaded {len(dataset)} subjects.")
    torch.save(dataset, "brain_node_labeled_dataset.pt")