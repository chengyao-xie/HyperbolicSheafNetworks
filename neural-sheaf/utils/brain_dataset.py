import os

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from utils.brain_graph import build_pyg_graph, pad_brain_graph_to_aal116
from utils.brain_io import (
    build_causality_caches,
    causality_cache_path,
    find_matrix_by_keyword,
    format_float_tag,
    load_causality_features,
    load_mat_file,
    save_causality_cache,
)
from utils.continuity_scaling import (
    ACX_DELAY,
    ACX_EMBEDDING_DIM,
    ACX_EPSILON_MAX,
    ACX_EPSILON_MIN,
    ACX_FEATURE_MODE,
    ACX_SCALE_BANDS,
    ACX_SCALES,
    ACX_SYMMETRIZATION,
    ACX_THEILER_WINDOW,
    continuity_scaling_multiscale_matrices,
)
from utils.simplicial import boundary_to_incidence_data


if os.environ.get('BRAIN_DATA_ROOT'):
    DEFAULT_DATA_ROOT = os.environ['BRAIN_DATA_ROOT']
elif os.name == 'nt':
    DEFAULT_DATA_ROOT = r'D:\BrainExp\data\data_process\data-driven-AAL'
else:
    DEFAULT_DATA_ROOT = os.path.join(
        os.path.expanduser('~'), 'code', 'dataset', 'data-driven-AAL')

DATASET_PATHS = {
    name: os.path.join(DEFAULT_DATA_ROOT, name)
    for name in ('abide', 'ppmi', 'taowu', 'neurocon', 'Matai')
}


def infer_graph_label(subject):
    subject = subject.lower()
    if 'control' in subject or 'baseline' in subject:
        return 0, 'control'
    if 'patient' in subject or 'postseason' in subject:
        return 1, 'patient'
    if 'prodromal' in subject:
        return 2, 'prodromal'
    if 'swedd' in subject:
        return 3, 'swedd'
    return None, None


def subject_matrix_path(subj_dir, construction_method):
    if construction_method == 'pearson':
        return find_matrix_by_keyword(subj_dir, 'correlation')
    cache_path = causality_cache_path(subj_dir)
    return cache_path if os.path.exists(cache_path) else find_matrix_by_keyword(
        subj_dir, 'features_timeseries')


def load_subject_graph_data(subj_dir, construction_method):
    data_path = subject_matrix_path(subj_dir, construction_method)
    if data_path is None:
        return None
    matrix = load_mat_file(data_path)
    if construction_method != 'causality':
        return matrix, None, None, None
    if matrix.shape[0] == matrix.shape[1]:
        reliability, scores, multiscale_reliability = load_causality_features(
            data_path, matrix.shape)
        return matrix, reliability, scores, multiscale_reliability

    matrix, reliability, scores, multiscale_reliability = (
        continuity_scaling_multiscale_matrices(matrix))
    save_causality_cache(
        causality_cache_path(subj_dir),
        matrix,
        reliability,
        scores,
        multiscale_reliability,
    )
    return matrix, reliability, scores, multiscale_reliability


def brain_graph_dataset(root_dir, top_k_percent=20, max_triangles=500,
                        simplex_order=2, construction_method='causality',
                        preprocess_workers=0):
    subjects = sorted(
        subject for subject in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, subject)))
    if construction_method == 'causality':
        build_causality_caches(root_dir, subjects, preprocess_workers)

    data_list = []
    counts = {'control': 0, 'patient': 0, 'prodromal': 0, 'swedd': 0}
    skipped = 0
    print(f"Found {len(subjects)} subjects to process...")
    for index, subject in enumerate(subjects, 1):
        if index == 1 or index % 10 == 0:
            print(f"Processing {index}/{len(subjects)}: {subject}")
        graph_label, label_type = infer_graph_label(subject)
        subject_data = load_subject_graph_data(
            os.path.join(root_dir, subject), construction_method)
        if graph_label is None or subject_data is None:
            skipped += 1
            continue

        matrix, reliability, scores, multiscale_reliability = subject_data
        data = build_pyg_graph(
            matrix,
            graph_label,
            top_k_percent,
            max_triangles,
            simplex_order,
            construction_method=construction_method,
            causality_reliability=reliability,
            multiscale_scores=scores,
            multiscale_reliability=multiscale_reliability,
        )
        if data is None:
            skipped += 1
            continue
        counts[label_type] += 1
        data_list.append(pad_brain_graph_to_aal116(data))

    print(f"Processed {len(data_list)} samples (skipped {skipped}):")
    print(f"  Control={counts['control']}, Patient={counts['patient']}, "
          f"Prodromal={counts['prodromal']}, SWEDD={counts['swedd']}")
    return data_list


class BrainGraphDataset:
    def __init__(self, pt_path, dataset_name):
        self.data, self.slices = torch.load(pt_path, weights_only=False)
        self.dataset_name = self.name = dataset_name
        sigma_path = pt_path.replace('.pt', '_sigma.pt')
        self.sigma_list = (
            torch.load(sigma_path, weights_only=False)
            if os.path.exists(sigma_path) else None)

    def __len__(self):
        return len(self.slices[next(iter(self.slices))]) - 1

    def __getitem__(self, idx):
        data = Data()
        for key, slices in self.slices.items():
            start, end = int(slices[idx]), int(slices[idx + 1])
            data[key] = self.data[key][:, start:end] if key == 'edge_index' else self.data[key][start:end]
        data.graph_idx = torch.tensor(idx, dtype=torch.long)

        if self.sigma_list is not None and idx < len(self.sigma_list) and self.sigma_list[idx] is not None:
            sigma = self.sigma_list[idx]
            data.sigma1, data.sigma2 = sigma['sigma1'], sigma['sigma2']
            if 'edge_node_index' in sigma:
                data.edge_node_index, data.edge_node_sign = (
                    sigma['edge_node_index'], sigma['edge_node_sign'])
                data.face_edge_index, data.face_edge_sign = (
                    sigma['face_edge_index'], sigma['face_edge_sign'])
            else:
                data.edge_node_index, data.edge_node_sign = boundary_to_incidence_data(data.sigma1)
                data.face_edge_index, data.face_edge_sign = boundary_to_incidence_data(data.sigma2)
            data.num_edges, data.num_triangles = sigma['num_edges'], sigma['num_triangles']
        return data

    @property
    def num_classes(self):
        return 4 if self.dataset_name.lower() == 'ppmi' else 2

    @property
    def num_features(self):
        return self.data.x.size(1)


def dataset_cache_name(dataset_name, construction_method, simplex_order, top_k_percent, max_triangles):
    triangle_tag = 'all' if max_triangles is None else str(max_triangles)
    acx_tag = ''
    if construction_method == 'causality':
        acx_tag = (
            f'_cs{ACX_EMBEDDING_DIM}_delay{ACX_DELAY}_scales{ACX_SCALES}'
            f'_tw{ACX_THEILER_WINDOW}'
            f'_eps{format_float_tag(ACX_EPSILON_MIN)}-{format_float_tag(ACX_EPSILON_MAX)}'
            f'_acxv5multiscale{ACX_SCALE_BANDS}_sym{ACX_SYMMETRIZATION}'
            f'_feat{ACX_FEATURE_MODE}'
        )
    return (
        f'{dataset_name}_{construction_method}_order{simplex_order}'
        f'_top{top_k_percent}_tri{triangle_tag}{acx_tag}.pt')


def save_dataset(data_list, pt_path):
    sigma_list = []
    for data in data_list:
        if not hasattr(data, 'sigma1') or data.sigma1 is None:
            sigma_list.append(None)
            continue
        sigma_list.append({
            'sigma1': data.sigma1,
            'sigma2': data.sigma2,
            'edge_node_index': data.edge_node_index,
            'edge_node_sign': data.edge_node_sign,
            'face_edge_index': data.face_edge_index,
            'face_edge_sign': data.face_edge_sign,
            'num_edges': int(data.num_edges),
            'num_triangles': int(data.num_triangles),
        })
        del (
            data.sigma1, data.sigma2,
            data.edge_node_index, data.edge_node_sign,
            data.face_edge_index, data.face_edge_sign,
        )

    torch.save(InMemoryDataset.collate(data_list), pt_path)
    if any(sigma is not None for sigma in sigma_list):
        torch.save(sigma_list, pt_path.replace('.pt', '_sigma.pt'))


def preprocess_dataset(dataset_name, top_k_percent=20, output_dir=None,
                       max_triangles=500, simplex_order=2, construction_method='causality',
                       preprocess_workers=0):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    pt_path = os.path.join(output_dir, dataset_cache_name(
        dataset_name, construction_method, simplex_order, top_k_percent, max_triangles))
    save_dataset(brain_graph_dataset(
        DATASET_PATHS[dataset_name],
        top_k_percent,
        max_triangles,
        simplex_order,
        construction_method,
        preprocess_workers,
    ), pt_path)
    print(f"Saved to {pt_path}")
    return pt_path


def get_dataset(name, simplex_order=2, top_k_percent=20,
                max_triangles=500, construction_method='causality',
                preprocess_workers=0):
    dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
    pt_path = os.path.join(dataset_dir, dataset_cache_name(
        name, construction_method, simplex_order, top_k_percent, max_triangles))
    if not os.path.exists(pt_path):
        pt_path = preprocess_dataset(
            name, top_k_percent, dataset_dir, max_triangles, simplex_order,
            construction_method, preprocess_workers)
    return BrainGraphDataset(pt_path, name)
