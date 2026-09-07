import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import scipy.io
import torch
from torch_geometric.data import Data, InMemoryDataset

from utils.continuity_scaling import continuity_scaling_matrix, normalize_signal_matrix
from utils.simplicial import (
    edge_to_node_matrix,
    hodge_2skeleton,
    triangle_to_edge_matrix,
)


DEFAULT_DATA_ROOT = os.environ.get('BRAIN_DATA_ROOT',
    r'D:\BrainExp\data\data_process\data-driven-AAL',
)

DATASET_PATHS = {
    'abide': os.path.join(DEFAULT_DATA_ROOT, 'abide'),
    'ppmi': os.path.join(DEFAULT_DATA_ROOT, 'ppmi'),
    'taowu': os.path.join(DEFAULT_DATA_ROOT, 'taowu'),
    'neurocon': os.path.join(DEFAULT_DATA_ROOT, 'neurocon'),
    'Matai': os.path.join(DEFAULT_DATA_ROOT, 'Matai'),
}


def load_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    for key in mat:
        if not key.startswith('__'):
            return mat[key]
    return None


def save_mat_file(file_path, data_matrix):
    scipy.io.savemat(file_path, {'data': data_matrix})


# label
def infer_graph_label(subject):
    subject_lower = subject.lower()
    if 'control' in subject_lower or 'baseline' in subject_lower:
        return 0, 'control'
    if 'patient' in subject_lower or 'postseason' in subject_lower:
        return 1, 'patient'
    if 'prodromal' in subject_lower:
        return 2, 'prodromal'
    if 'swedd' in subject_lower:
        return 3, 'swedd'
    return None, None


def find_matrix_by_keyword(subj_dir, keywords):
    if isinstance(keywords, str):
        keywords = [keywords]
    keyword_list = [keyword.lower() for keyword in keywords]
    for filename in os.listdir(subj_dir):
        filename_lower = filename.lower()
        if any(keyword in filename_lower for keyword in keyword_list):
            return os.path.join(subj_dir, filename)
    return None


#  持续缩放处理
def causality_cache_path(subj_dir, cs_embedding_dim=3, cs_delay=1, cs_scales=33,
                         cs_theiler_window=2, cs_epsilon_min=0.001,
                         cs_epsilon_max=1.0):
    subject = os.path.basename(subj_dir.rstrip(os.sep))
    return os.path.join(subj_dir, f'{subject}_AAL116_causality_matrix.mat')


# 不同方式构造脑网络
def find_subject_matrix(subj_dir, construction_method, cs_embedding_dim=3, cs_delay=1, cs_scales=33,
                        cs_theiler_window=2, cs_epsilon_min=0.001,
                        cs_epsilon_max=1.0):
    if construction_method == 'causality':
        cache_path = causality_cache_path(
            subj_dir,
            cs_embedding_dim=cs_embedding_dim,
            cs_delay=cs_delay,
            cs_scales=cs_scales,
            cs_theiler_window=cs_theiler_window,
            cs_epsilon_min=cs_epsilon_min,
            cs_epsilon_max=cs_epsilon_max,
        )
        if os.path.exists(cache_path):
            return cache_path
        return find_matrix_by_keyword(subj_dir, 'features_timeseries')
    return find_matrix_by_keyword(subj_dir, 'correlation')


def build_causality_cache_for_subject(task):
    (subj_dir, cs_embedding_dim, cs_delay, cs_scales, cs_theiler_window,
     cs_epsilon_min, cs_epsilon_max) = task
    subject = os.path.basename(subj_dir.rstrip(os.sep))
    cache_path = causality_cache_path(
        subj_dir,
        cs_embedding_dim=cs_embedding_dim,
        cs_delay=cs_delay,
        cs_scales=cs_scales,
        cs_theiler_window=cs_theiler_window,
        cs_epsilon_min=cs_epsilon_min,
        cs_epsilon_max=cs_epsilon_max,
    )
    if os.path.exists(cache_path):
        return subject, cache_path, 'cached', None

    data_path = find_matrix_by_keyword(subj_dir, 'features_timeseries')
    if data_path is None:
        return subject, None, 'skipped', 'features_timeseries file not found'

    try:
        data_matrix = load_mat_file(data_path)
        if data_matrix is None:
            return subject, None, 'skipped', 'empty mat file'
        if data_matrix.shape[0] == data_matrix.shape[1]:
            save_mat_file(cache_path, data_matrix)
        else:
            causality_matrix = continuity_scaling_matrix(
                data_matrix,
                embedding_dim=cs_embedding_dim,
                delay=cs_delay,
                scale_count=cs_scales,
                theiler_window=cs_theiler_window,
                epsilon_min=cs_epsilon_min,
                epsilon_max=cs_epsilon_max,
            )
            save_mat_file(cache_path, causality_matrix)
    except Exception as exc:
        return subject, None, 'failed', str(exc)

    return subject, cache_path, 'built', None


def ensure_causality_caches(root_dir, subjects, workers=0, cs_embedding_dim=3, cs_delay=1,
                            cs_scales=33, cs_theiler_window=2, cs_epsilon_min=0.001,
                            cs_epsilon_max=1.0):
    tasks = []
    for subject in subjects:
        subj_dir = os.path.join(root_dir, subject)
        cache_path = causality_cache_path(
            subj_dir,
            cs_embedding_dim=cs_embedding_dim,
            cs_delay=cs_delay,
            cs_scales=cs_scales,
            cs_theiler_window=cs_theiler_window,
            cs_epsilon_min=cs_epsilon_min,
            cs_epsilon_max=cs_epsilon_max,
        )
        if not os.path.exists(cache_path):
            tasks.append((
                subj_dir,
                cs_embedding_dim,
                cs_delay,
                cs_scales,
                cs_theiler_window,
                cs_epsilon_min,
                cs_epsilon_max,
            ))

    if not tasks:
        print("Causality matrix cache complete; no subjects to build.")
        return

    if workers <= 1:
        return

    workers = min(workers, len(tasks))
    print(f"Building {len(tasks)} causality matrices with {workers} workers...")
    built = failed = skipped = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(build_causality_cache_for_subject, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            subject, cache_path, status, error = future.result()
            if status == 'built':
                built += 1
            elif status == 'failed':
                failed += 1
                print(f"  Warning: Failed causality cache for {subject}: {error}")
            else:
                skipped += 1
            if done == 1 or done % 10 == 0 or done == len(futures):
                print(f"  Causality cache {done}/{len(futures)} | built={built} | "
                      f"cached/skipped={skipped} | failed={failed}")


# 根据连接强度构建阈值化邻接矩阵。
def build_thresholded_adjacency(connectivity_matrix, top_k_percent):
    n_nodes = connectivity_matrix.shape[0]
    connectivities = []
    positions = []

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            value = connectivity_matrix[i, j]
            connectivities.append(abs(value))
            positions.append((i, j, value))

    if not connectivities:
        return None

    threshold = np.percentile(connectivities, 100 - top_k_percent)
    adj_binary = np.zeros((n_nodes, n_nodes))
    for i, j, value in positions:
        if abs(value) >= threshold:
            adj_binary[i, j] = 1
            adj_binary[j, i] = 1

    return adj_binary


# 生成连接矩阵和节点特征。
def build_connectivity_and_features(data_matrix, construction_method='causality',
                                    cs_embedding_dim=3, cs_delay=1, cs_scales=33,
                                    cs_theiler_window=2, cs_epsilon_min=0.001,
                                    cs_epsilon_max=1.0):
    matrix = np.asarray(data_matrix, dtype=float)
    if construction_method == 'causality':
        if matrix.shape[0] == matrix.shape[1]:
            connectivity = matrix
        else:
            connectivity = continuity_scaling_matrix(
                matrix,
                embedding_dim=cs_embedding_dim,
                delay=cs_delay,
                scale_count=cs_scales,
                theiler_window=cs_theiler_window,
                epsilon_min=cs_epsilon_min,
                epsilon_max=cs_epsilon_max,
            )
        features = connectivity
    elif construction_method == 'pearson' and matrix.shape[0] != matrix.shape[1]:
        roi_timeseries = normalize_signal_matrix(matrix)
        connectivity = np.corrcoef(roi_timeseries)
        connectivity = np.nan_to_num(connectivity, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(connectivity, 0.0)
        features = connectivity
    else:
        connectivity = matrix
        features = matrix
    return connectivity, features


def build_sparse_hodge_operators(sigma1, sigma2):
    """Build constant sparse operators used by the order-2 diffusion."""
    sigma1 = sigma1.coalesce() if sigma1.is_sparse else sigma1.to_sparse().coalesce()
    sigma2 = sigma2.coalesce() if sigma2.is_sparse else sigma2.to_sparse().coalesce()

    lower = torch.sparse.mm(sigma1.t(), sigma1)
    if sigma2.size(1) > 0:
        upper = torch.sparse.mm(sigma2, sigma2.t())
        hodge_laplacian = (lower + upper).coalesce()
    else:
        hodge_laplacian = lower.coalesce()
    return sigma1, hodge_laplacian


# 按权重保留最重要的三角形。
def limit_triangles_by_weight(triangles, connectivity_matrix, max_triangles):
    if max_triangles is None or len(triangles) <= max_triangles:
        return triangles

    tri_weights = []
    for tri in triangles:
        i, j, k = tri[0] - 1, tri[1] - 1, tri[2] - 1
        weight = abs(connectivity_matrix[i, j]) + abs(connectivity_matrix[j, k]) + abs(connectivity_matrix[i, k])
        tri_weights.append(weight)

    sorted_indices = np.argsort(tri_weights)[::-1][:max_triangles]
    return [triangles[i] for i in sorted_indices]


# 构建 PyG 图样本。
def build_pyg_graph(data_matrix, graph_label, top_k_percent, max_triangles, simplex_order,
                    construction_method='pearson', cs_embedding_dim=3, cs_delay=1,
                    cs_scales=33, cs_theiler_window=2, cs_epsilon_min=0.001,
                    cs_epsilon_max=1.0):
    connectivity, features = build_connectivity_and_features(
        data_matrix,
        construction_method=construction_method,
        cs_embedding_dim=cs_embedding_dim,
        cs_delay=cs_delay,
        cs_scales=cs_scales,
        cs_theiler_window=cs_theiler_window,
        cs_epsilon_min=cs_epsilon_min,
        cs_epsilon_max=cs_epsilon_max,
    )
    adj_binary = build_thresholded_adjacency(connectivity, top_k_percent)
    if adj_binary is None:
        return None

    nodes, edges, triangles = hodge_2skeleton(adj_binary)
    nodes = nodes.flatten().tolist()
    edges = edges.tolist()
    triangles = triangles.tolist()

    if not edges:
        return None

    if simplex_order == 2:
        triangles = limit_triangles_by_weight(triangles, connectivity, max_triangles)
        sigma1 = edge_to_node_matrix(edges, nodes, one_indexed=True)
        sigma2 = (
            triangle_to_edge_matrix(triangles, edges)
            if triangles else torch.zeros((len(edges), 0), dtype=torch.float)
        )
        sigma1, hodge_laplacian = build_sparse_hodge_operators(sigma1, sigma2)
    else:
        sigma1 = None
        hodge_laplacian = None
        triangles = []

    edge_list = []
    edge_weights = []
    for edge in edges:
        i, j = edge[0] - 1, edge[1] - 1
        value = connectivity[i, j]
        edge_list.append([i, j])
        edge_list.append([j, i])
        edge_weights.extend([value, value])

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float32).view(-1, 1),
        y=torch.tensor(graph_label, dtype=torch.long),
    )

    if simplex_order == 2:
        data.sigma1 = sigma1
        data.hodge_laplacian = hodge_laplacian
        data.num_triangles = len(triangles)
        data.num_edges = len(edges)

    return data


# 构建脑图数
def brain_graph_dataset(root_dir, top_k_percent=20, max_triangles=None,
                        simplex_order=2, construction_method='pearson', cs_embedding_dim=3,
                        cs_delay=1, cs_scales=33, cs_theiler_window=2,
                        cs_epsilon_min=0.001, cs_epsilon_max=1.0,
                        preprocess_workers=0):
    data_list = []
    subjects = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    counts = {'control': 0, 'patient': 0, 'prodromal': 0, 'swedd': 0}
    skipped = 0

    print(f"Found {len(subjects)} subjects to process...")

    if construction_method == 'causality':
        ensure_causality_caches(
            root_dir,
            subjects,
            workers=preprocess_workers,
            cs_embedding_dim=cs_embedding_dim,
            cs_delay=cs_delay,
            cs_scales=cs_scales,
            cs_theiler_window=cs_theiler_window,
            cs_epsilon_min=cs_epsilon_min,
            cs_epsilon_max=cs_epsilon_max,
        )

    for idx, subject in enumerate(subjects, 1):
        if idx % 10 == 0 or idx == 1:
            print(f"Processing {idx}/{len(subjects)}: {subject}")

        subj_dir = os.path.join(root_dir, subject)
        data_path = find_subject_matrix(
            subj_dir,
            construction_method,
            cs_embedding_dim=cs_embedding_dim,
            cs_delay=cs_delay,
            cs_scales=cs_scales,
            cs_theiler_window=cs_theiler_window,
            cs_epsilon_min=cs_epsilon_min,
            cs_epsilon_max=cs_epsilon_max,
        )
        if data_path is None:
            skipped += 1
            continue

        try:
            data_matrix = load_mat_file(data_path)
        except Exception as exc:
            print(f"  Warning: Failed to load {subject}: {exc}")
            skipped += 1
            continue

        if data_matrix is None:
            skipped += 1
            continue

        if construction_method == 'causality' and data_matrix.shape[0] != data_matrix.shape[1]:
            cache_path = causality_cache_path(
                subj_dir,
                cs_embedding_dim=cs_embedding_dim,
                cs_delay=cs_delay,
                cs_scales=cs_scales,
                cs_theiler_window=cs_theiler_window,
                cs_epsilon_min=cs_epsilon_min,
                cs_epsilon_max=cs_epsilon_max,
            )
            try:
                data_matrix = continuity_scaling_matrix(
                    data_matrix,
                    embedding_dim=cs_embedding_dim,
                    delay=cs_delay,
                    scale_count=cs_scales,
                    theiler_window=cs_theiler_window,
                    epsilon_min=cs_epsilon_min,
                    epsilon_max=cs_epsilon_max,
                )
                save_mat_file(cache_path, data_matrix)
                print(f"  Saved causality matrix: {cache_path}")
            except Exception as exc:
                print(f"  Warning: Failed to build causality matrix for {subject}: {exc}")
                skipped += 1
                continue

        graph_label, label_type = infer_graph_label(subject)
        if graph_label is None:
            continue

        counts[label_type] += 1
        data = build_pyg_graph(
            data_matrix,
            graph_label,
            top_k_percent,
            max_triangles,
            simplex_order,
            construction_method=construction_method,
            cs_embedding_dim=cs_embedding_dim,
            cs_delay=cs_delay,
            cs_scales=cs_scales,
            cs_theiler_window=cs_theiler_window,
            cs_epsilon_min=cs_epsilon_min,
            cs_epsilon_max=cs_epsilon_max,
        )
        if data is None:
            skipped += 1
            continue

        data_list.append(data)

        if idx % 50 == 0:
            import gc
            gc.collect()

    print(f"Processed {len(data_list)} samples (skipped {skipped}):")
    print(f"  Control={counts['control']}, Patient={counts['patient']}, "
          f"Prodromal={counts['prodromal']}, SWEDD={counts['swedd']}")
    return data_list


class BrainGraphDataset:
    # 初始化 BrainGraphDataset 对象。
    def __init__(self, pt_path, dataset_name):
        self.data, self.slices = torch.load(pt_path, weights_only=False)
        self.dataset_name = dataset_name
        self.name = dataset_name
        sigma_path = pt_path.replace('.pt', '_sigma.pt')
        self.sigma_list = (
            torch.load(sigma_path, weights_only=False)
            if os.path.exists(sigma_path) else None
        )

    # 返回样本数量。
    def __len__(self):
        return len(self.slices[list(self.slices.keys())[0]]) - 1

    # 按索引读取单个图样本。
    def __getitem__(self, idx):
        data = Data()
        for key in self.slices.keys():
            start = int(self.slices[key][idx])
            end = int(self.slices[key][idx + 1])

            if key == 'edge_index':
                data[key] = self.data[key][:, start:end]
            else:
                data[key] = self.data[key][start:end]

        data.graph_idx = torch.tensor(idx, dtype=torch.long)

        if (self.sigma_list is not None
                and idx < len(self.sigma_list)
                and self.sigma_list[idx] is not None):
            sigma_data = self.sigma_list[idx]
            if 'hodge_laplacian' not in sigma_data:
                # Upgrade legacy companions lazily and retain the result in
                # memory. This one-time conversion replaces thousands of dense
                # Hodge reconstructions during training.
                sigma1, hodge_laplacian = build_sparse_hodge_operators(
                    sigma_data['sigma1'], sigma_data['sigma2'])
                sigma_data['sigma1'] = sigma1
                sigma_data['hodge_laplacian'] = hodge_laplacian
                sigma_data.pop('sigma2', None)
            data.sigma1 = sigma_data['sigma1']
            data.hodge_laplacian = sigma_data['hodge_laplacian']
            data.num_edges = sigma_data['num_edges']
            data.num_triangles = sigma_data['num_triangles']

        return data

    # 返回类别数。
    @property
    def num_classes(self):
        return 4 if self.dataset_name.lower() == "ppmi" else 2

    # 返回节点特征维度。
    @property
    def num_features(self):
        return self.data.x.size(1)


def format_float_tag(value):
    return f"{value:g}".replace('.', 'p').replace('-', 'm')


# 预处理缓存
def dataset_cache_name(dataset_name, construction_method, simplex_order, top_k_percent, max_triangles,
                       cs_embedding_dim=3, cs_delay=1, cs_scales=33, cs_theiler_window=2,
                       cs_epsilon_min=0.001, cs_epsilon_max=1.0):
    triangle_tag = 'all' if max_triangles is None else str(max_triangles)
    if construction_method == 'causality':
        cs_tag = (
            f"_cs{cs_embedding_dim}_delay{cs_delay}_scales{cs_scales}"
            f"_tw{cs_theiler_window}"
            f"_eps{format_float_tag(cs_epsilon_min)}-{format_float_tag(cs_epsilon_max)}"
        )
    else:
        cs_tag = ''
    return (
        f"{dataset_name}_{construction_method}_order{simplex_order}"
        f"_top{top_k_percent}_tri{triangle_tag}{cs_tag}.pt"
    )


def legacy_dataset_cache_path(dataset_dir, dataset_name, construction_method, simplex_order,
                              top_k_percent, max_triangles, cs_embedding_dim=3, cs_delay=1,
                              cs_scales=33, cs_theiler_window=2, cs_epsilon_min=0.001,
                              cs_epsilon_max=1.0):
    """Return a compatible pre-parameterised cache path when one exists."""
    # Legacy filenames omitted every graph-construction parameter, so only
    # reuse them for the defaults under which those files were produced.
    default_graph = top_k_percent == 20 and max_triangles == 500
    default_cs = (
        cs_embedding_dim == 3 and cs_delay == 1 and cs_scales == 33
        and cs_theiler_window == 2 and cs_epsilon_min == 0.001
        and cs_epsilon_max == 1.0
    )
    if not default_graph or (construction_method == 'causality' and not default_cs):
        return None

    legacy_name = f"{dataset_name}_{construction_method}_order{simplex_order}.pt"
    legacy_path = os.path.join(dataset_dir, legacy_name)
    return legacy_path if os.path.exists(legacy_path) else None


# 预处理
def preprocess_dataset(dataset_name, top_k_percent=20, output_dir=None,
                       max_triangles=500, simplex_order=2, construction_method='pearson',
                       cs_embedding_dim=3, cs_delay=1, cs_scales=33, cs_theiler_window=2,
                       cs_epsilon_min=0.001, cs_epsilon_max=1.0,
                       preprocess_workers=0):
    root_dir = DATASET_PATHS[dataset_name]

    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(os.path.dirname(current_dir), 'dataset')

    os.makedirs(output_dir, exist_ok=True)
    pt_filename = dataset_cache_name(
        dataset_name,
        construction_method,
        simplex_order,
        top_k_percent,
        max_triangles,
        cs_embedding_dim=cs_embedding_dim,
        cs_delay=cs_delay,
        cs_scales=cs_scales,
        cs_theiler_window=cs_theiler_window,
        cs_epsilon_min=cs_epsilon_min,
        cs_epsilon_max=cs_epsilon_max,
    )
    pt_path = os.path.join(output_dir, pt_filename)

    data_list = brain_graph_dataset(
        root_dir,
        top_k_percent,
        max_triangles=max_triangles,
        simplex_order=simplex_order,
        construction_method=construction_method,
        cs_embedding_dim=cs_embedding_dim,
        cs_delay=cs_delay,
        cs_scales=cs_scales,
        cs_theiler_window=cs_theiler_window,
        cs_epsilon_min=cs_epsilon_min,
        cs_epsilon_max=cs_epsilon_max,
        preprocess_workers=preprocess_workers,
    )
    if not data_list:
        raise RuntimeError(
            f"No samples were built for dataset={dataset_name}, construction_method={construction_method}."
        )

    sigma_list = []
    for data in data_list:
        if hasattr(data, 'sigma1') and data.sigma1 is not None:
            sigma_list.append({
                'sigma1': data.sigma1,
                'hodge_laplacian': data.hodge_laplacian,
                'num_edges': int(data.num_edges),
                'num_triangles': int(data.num_triangles),
            })
            del data.sigma1, data.hodge_laplacian
        else:
            sigma_list.append(None)

    data, slices = InMemoryDataset.collate(data_list)
    torch.save((data, slices), pt_path)

    if any(s is not None for s in sigma_list):
        sigma_path = pt_path.replace('.pt', '_sigma.pt')
        torch.save(sigma_list, sigma_path)
        print(f"  Saved sigma companion to {sigma_path}")

    print(f" Saved to {pt_path}\n")
    return pt_path


# 加载或生成预处理数据集
def get_dataset(name, simplex_order=2, top_k_percent=20,
                max_triangles=500, construction_method='pearson',
                cs_embedding_dim=3, cs_delay=1, cs_scales=33, cs_theiler_window=2,
                cs_epsilon_min=0.001, cs_epsilon_max=1.0,
                preprocess_workers=0):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(current_dir), 'dataset')
    pt_filename = dataset_cache_name(
        name,
        construction_method,
        simplex_order,
        top_k_percent,
        max_triangles,
        cs_embedding_dim=cs_embedding_dim,
        cs_delay=cs_delay,
        cs_scales=cs_scales,
        cs_theiler_window=cs_theiler_window,
        cs_epsilon_min=cs_epsilon_min,
        cs_epsilon_max=cs_epsilon_max,
    )
    pt_path = os.path.join(dataset_dir, pt_filename)

    if not os.path.exists(pt_path):
        legacy_path = legacy_dataset_cache_path(
            dataset_dir, name, construction_method, simplex_order,
            top_k_percent, max_triangles,
            cs_embedding_dim=cs_embedding_dim,
            cs_delay=cs_delay,
            cs_scales=cs_scales,
            cs_theiler_window=cs_theiler_window,
            cs_epsilon_min=cs_epsilon_min,
            cs_epsilon_max=cs_epsilon_max,
        )
        if legacy_path is not None:
            pt_path = legacy_path
            print(f"Using compatible legacy preprocessing cache: {pt_path}")
        else:
            print("Preprocessed file not found, generating...")
            pt_path = preprocess_dataset(
                name,
                top_k_percent=top_k_percent,
                max_triangles=max_triangles,
                simplex_order=simplex_order,
                construction_method=construction_method,
                cs_embedding_dim=cs_embedding_dim,
                cs_delay=cs_delay,
                cs_scales=cs_scales,
                cs_theiler_window=cs_theiler_window,
                cs_epsilon_min=cs_epsilon_min,
                cs_epsilon_max=cs_epsilon_max,
                preprocess_workers=preprocess_workers,
            )

    dataset = BrainGraphDataset(pt_path, name)
    dataset.name = name
    return dataset
