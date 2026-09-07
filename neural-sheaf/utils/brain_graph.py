import numpy as np
import torch
from torch_geometric.data import Data

from utils.continuity_scaling import (
    ACX_SCALE_BANDS,
    ACX_SYMMETRIZATION,
    normalize_signal_matrix,
)
from utils.simplicial import (
    boundary_to_incidence_data,
    edge_to_node_matrix,
    hodge_2skeleton,
    triangle_to_edge_matrix,
)


def build_thresholded_adjacency(connectivity_matrix, top_k_percent,
                                ensure_connected=False):
    n_nodes = connectivity_matrix.shape[0]
    positions = [
        (i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)
    ]
    if not positions:
        return None

    weights = np.asarray([abs(connectivity_matrix[i, j]) for i, j in positions])
    target_count = max(1, int(np.ceil(len(positions) * top_k_percent / 100.0)))
    if ensure_connected:
        target_count = max(target_count, n_nodes - 1)
    order = np.argsort(-weights, kind='mergesort')
    selected = set()

    if ensure_connected:
        parent = np.arange(n_nodes)

        def find(node):
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for index in order:
            left, right = positions[index]
            left, right = find(left), find(right)
            if left != right:
                parent[right] = left
                selected.add(index)
                if len(selected) == n_nodes - 1:
                    break

    for index in order:
        if len(selected) >= target_count:
            break
        selected.add(index)
    adjacency = np.zeros((n_nodes, n_nodes))
    for index in selected:
        left, right = positions[index]
        adjacency[left, right] = adjacency[right, left] = 1
    return adjacency


def normalise_features(scores):
    features = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    off_diagonal = np.abs(features[~np.eye(features.shape[0], dtype=bool)])
    scale = np.percentile(off_diagonal, 95) if off_diagonal.size else 0.0
    features = np.clip(features / scale, -3.0, 3.0) if scale > 1e-12 else np.zeros_like(features)
    np.fill_diagonal(features, 0.0)
    return features


def causality_topology(scores, symmetrization):
    strengths = np.abs(scores)
    if symmetrization == 'max':
        strengths = np.maximum(strengths, strengths.T)
    else:
        strengths = 0.5 * (strengths + strengths.T)
    np.fill_diagonal(strengths, 0.0)
    return np.nan_to_num(strengths, nan=0.0, posinf=0.0, neginf=0.0)


def build_connectivity(matrix, construction_method):
    matrix = np.asarray(matrix, dtype=float)
    if construction_method == 'causality':
        return causality_topology(matrix, ACX_SYMMETRIZATION)
    if matrix.shape[0] != matrix.shape[1]:
        matrix = np.corrcoef(normalize_signal_matrix(matrix))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def multiscale_features(scores, symmetrization):
    return np.concatenate([
        normalise_features(causality_topology(scores[..., band], symmetrization))
        for band in range(scores.shape[-1])
    ], axis=1)


def limit_triangles(triangles, connectivity, max_triangles):
    if max_triangles is None or len(triangles) <= max_triangles:
        return triangles
    weights = [
        sum(abs(connectivity[a - 1, b - 1]) for a, b in ((i, j), (j, k), (i, k)))
        for i, j, k in triangles
    ]
    return [triangles[index] for index in np.argsort(weights)[::-1][:max_triangles]]


def acx_node_features(scores, multiscale_scores, reliability, multiscale_reliability):
    strength_blocks, reliability_blocks = [], []
    for symmetrization in ('mean', 'max'):
        strength_blocks.extend((
            normalise_features(causality_topology(scores, symmetrization)),
            multiscale_features(multiscale_scores, symmetrization),
        ))
        reliability_blocks.append(causality_topology(reliability, symmetrization))
        reliability_blocks.extend(
            causality_topology(multiscale_reliability[..., band], symmetrization)
            for band in range(multiscale_reliability.shape[-1])
        )
    return np.concatenate([*strength_blocks, *reliability_blocks], axis=1)


def edge_features(connectivity, construction_method, scores, reliability):
    edge_scale = np.max(np.abs(connectivity))
    edge_scale = edge_scale if np.isfinite(edge_scale) and edge_scale > 1e-12 else 1.0
    if construction_method == 'causality':
        strengths = np.stack([
            normalise_features(scores[..., band])
            for band in range(scores.shape[-1])
        ], axis=-1)
        reliability = np.clip(np.nan_to_num(reliability), 0.0, 1.0)
    else:
        strengths = np.repeat(
            (np.abs(connectivity) / edge_scale)[..., None], ACX_SCALE_BANDS, axis=-1)
        reliability = np.ones_like(strengths)
    return edge_scale, strengths, reliability


def build_pyg_graph(matrix, graph_label, top_k_percent, max_triangles, simplex_order,
                    construction_method='causality', causality_reliability=None,
                    multiscale_scores=None, multiscale_reliability=None):
    connectivity = build_connectivity(matrix, construction_method)
    features = connectivity
    if construction_method == 'causality':
        scores = np.asarray(matrix, dtype=float)
        multiscale_scores = (
            multiscale_scores if multiscale_scores is not None
            else np.repeat(scores[..., None], ACX_SCALE_BANDS, axis=-1))
        aggregate_reliability = (
            causality_reliability if causality_reliability is not None
            else np.ones_like(scores))
        multiscale_reliability = (
            multiscale_reliability if multiscale_reliability is not None
            else np.ones_like(multiscale_scores))
        features = acx_node_features(
            scores, multiscale_scores, aggregate_reliability, multiscale_reliability)
        edge_reliability = multiscale_reliability
    else:
        edge_reliability = None

    adjacency = build_thresholded_adjacency(
        connectivity, top_k_percent, ensure_connected=construction_method == 'causality')
    if adjacency is None:
        return None
    nodes, edges, triangles = hodge_2skeleton(adjacency)
    nodes, edges, triangles = nodes.flatten().tolist(), edges.tolist(), triangles.tolist()
    if not edges:
        return None

    if simplex_order == 2:
        triangles = limit_triangles(triangles, connectivity, max_triangles)
        sigma1 = edge_to_node_matrix(edges, nodes, one_indexed=True)
        sigma2 = triangle_to_edge_matrix(triangles, edges) if triangles else torch.zeros((len(edges), 0))
        edge_node_index, edge_node_sign = boundary_to_incidence_data(sigma1)
        face_edge_index, face_edge_sign = boundary_to_incidence_data(sigma2)

    edge_scale, strengths, reliability = edge_features(
        connectivity, construction_method, multiscale_scores, edge_reliability)
    edge_index, edge_attr, sheaf_edge_attr = [], [], []
    for left, right in edges:
        left, right = left - 1, right - 1
        edge_index.extend(((left, right), (right, left)))
        edge_attr.extend((connectivity[left, right] / edge_scale,) * 2)
        sheaf_edge_attr.extend((
            np.concatenate((strengths[right, left], reliability[right, left])),
            np.concatenate((strengths[left, right], reliability[left, right])),
        ))

    data = Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32).view(-1, 1),
        sheaf_edge_attr=torch.tensor(np.asarray(sheaf_edge_attr), dtype=torch.float32),
        y=torch.tensor(graph_label, dtype=torch.long),
    )
    if simplex_order == 2:
        data.sigma1, data.sigma2 = sigma1, sigma2
        data.edge_node_index, data.edge_node_sign = edge_node_index, edge_node_sign
        data.face_edge_index, data.face_edge_sign = face_edge_index, face_edge_sign
        data.num_edges, data.num_triangles = len(edges), len(triangles)
    return data


def pad_brain_graph_to_aal116(data, target_nodes=116):
    node_count = int(data.x.size(0))
    if node_count == target_nodes:
        return data
    block_count = data.x.size(1) // node_count
    source = data.x.view(node_count, block_count, node_count)
    padded = data.x.new_zeros((target_nodes, block_count, target_nodes))
    padded[:node_count, :, :node_count] = source
    data.x = padded.reshape(target_nodes, block_count * target_nodes)
    if hasattr(data, 'sigma1') and data.sigma1 is not None:
        sigma1 = data.sigma1
        if sigma1.layout == torch.strided:
            padded_sigma1 = sigma1.new_zeros((target_nodes, sigma1.size(1)))
            padded_sigma1[:node_count] = sigma1
        else:
            sigma1 = sigma1.to_sparse_coo().coalesce()
            padded_sigma1 = torch.sparse_coo_tensor(
                sigma1.indices(), sigma1.values(), (target_nodes, sigma1.size(1)),
                dtype=sigma1.dtype, device=sigma1.device).coalesce()
        data.sigma1 = padded_sigma1
        data.edge_node_index, data.edge_node_sign = boundary_to_incidence_data(data.sigma1)
    return data
