import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os.path as osp
import torch
import numpy as np
import scipy.io
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected, remove_self_loops, coalesce


class BrainNetworkDataset(InMemoryDataset):
    def __init__(self, root, name, threshold=0.7, transform=None, pre_transform=None):
        self.name = name.lower()
        self.threshold = threshold
        assert self.name in ['neurocon']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return []  # 不用检查存在文件名

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    @staticmethod
    def load_mat_file(file_path):
        mat = scipy.io.loadmat(file_path)
        for key in mat:
            if not key.startswith('__'):
                return mat[key]
        raise ValueError(f"No valid variable found in {file_path}")

    @staticmethod
    def find_file_with_keywords(directory, required_keywords):
        for f in os.listdir(directory):
            if f.endswith(".mat") and all(k.lower() in f.lower() for k in required_keywords):
                return osp.join(directory, f)
        return None

    def process(self):
        data_list = []
        subjects = sorted([d for d in os.listdir(self.raw_dir)
                           if osp.isdir(osp.join(self.raw_dir, d))])

        for subject in subjects:
            subj_dir = osp.join(self.raw_dir, subject)

            corr_path = self.find_file_with_keywords(subj_dir, ["correlation", "aal116"])
            ts_path = self.find_file_with_keywords(subj_dir, ["timeseries", "aal116"])

            print(f"\nProcessing {subject}")
            print(f"  corr_path: {corr_path}")
            print(f"  ts_path: {ts_path}")

            if corr_path is None or ts_path is None:
                print(f"  ✘ Missing files for {subject}, skipped.")
                continue

            try:
                corr = self.load_mat_file(corr_path)
                ts = self.load_mat_file(ts_path)

                if corr.shape != (116, 116) or ts.shape[1] != 116:
                    print(f"  ✘ Shape mismatch for {subject}, skipped.")
                    continue

                node_features = np.mean(ts, axis=0)
                x = torch.tensor(node_features, dtype=torch.float32).view(-1, 1)
                y = torch.tensor(0, dtype=torch.long)  # 默认label为0，可根据需要修改

                edge_index = []
                for i in range(116):
                    for j in range(i + 1, 116):
                        if abs(corr[i, j]) > self.threshold:
                            edge_index.append([i, j])
                            edge_index.append([j, i])

                if len(edge_index) == 0:
                    print(f"  ✘ No edges after thresholding, skipped.")
                    continue

                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_index = to_undirected(edge_index)
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

                data = Data(x=x, edge_index=edge_index, y=y)
                data.subject_id = subject
                data_list.append(data)
                print(f"  ✓ Added graph for {subject} with {edge_index.size(1)} edges.")

            except Exception as e:
                print(f"  ✘ Error processing {subject}: {e}")
                continue

        if len(data_list) == 0:
            raise RuntimeError("No valid data found. Check file names and data formats.")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name.capitalize()}BrainDataset(threshold={self.threshold})'

if __name__ == "__main__":
    dataset = BrainNetworkDataset(root='D:/脑网络', name='neurocon', threshold=0.7)
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of node features: {dataset.num_features}")

