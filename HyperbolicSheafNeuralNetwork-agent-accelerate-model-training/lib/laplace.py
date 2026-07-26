import torch


# 计算双向边中每条无向边的左右限制映射索引。
def compute_left_right_map_index(edge_index, full_matrix=False):
    """Match every directed edge with its reverse without host-device syncs."""
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return empty, empty

    row, col = edge_index
    if torch.any(row == col):
        raise ValueError("Self loops are not supported by the sheaf Laplacian builder")

    # Encode (source, target) as a scalar key, then find reverse keys with a
    # vectorised sort/search. The old Python dictionary implementation called
    # .item() several times per edge and synchronised CUDA on every lookup.
    num_nodes = edge_index.max() + 1
    edge_keys = row * num_nodes + col
    reverse_keys = col * num_nodes + row
    sorted_keys, permutation = torch.sort(edge_keys)
    reverse_positions = torch.searchsorted(sorted_keys, reverse_keys)

    if torch.any(reverse_positions >= sorted_keys.numel()):
        raise ValueError("edge_index must contain the reverse of every edge")
    matched_keys = sorted_keys.index_select(0, reverse_positions)
    if torch.any(matched_keys != reverse_keys):
        raise ValueError("edge_index must contain the reverse of every edge")

    reverse_index = permutation.index_select(0, reverse_positions)
    if full_matrix:
        left_index = torch.arange(edge_index.size(1), device=edge_index.device)
    else:
        left_index = torch.nonzero(row < col, as_tuple=False).view(-1)
        if left_index.numel() * 2 != edge_index.size(1):
            raise ValueError("edge_index must describe a bidirectional simple graph")

    right_index = reverse_index.index_select(0, left_index)
    left_right_index = torch.stack([left_index, right_index], dim=0)
    return left_right_index, edge_index.index_select(1, left_index)


# 计算对角 sheaf Laplacian 中可学习部分的稀疏索引。
def compute_learnable_diag_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])
    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(0, learned_d, device=device).view(1, -1)
    col_template = row_template.clone()

    non_diag_row_indices = (row_template + total_d * row.unsqueeze(1)).reshape(1, -1)
    non_diag_col_indices = (col_template + total_d * col.unsqueeze(1)).reshape(1, -1)
    non_diag_indices = torch.cat((non_diag_row_indices, non_diag_col_indices), dim=0)

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d * diag.unsqueeze(1)).reshape(1, -1)
    diag_col_indices = (col_template + total_d * diag.unsqueeze(1)).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


# 计算固定低通/高通维度在对角 Laplacian 中的稀疏索引。
def compute_fixed_diag_laplacian_indices(size, edge_index, learned_d, total_d):
    assert torch.all(edge_index[0] < edge_index[1])
    row, col = edge_index
    device = edge_index.device
    row_template = torch.arange(learned_d, total_d, device=device).view(1, -1)
    col_template = row_template.clone()

    non_diag_row_indices = (row_template + total_d * row.unsqueeze(1)).reshape(1, -1)
    non_diag_col_indices = (col_template + total_d * col.unsqueeze(1)).reshape(1, -1)
    non_diag_indices = torch.cat((non_diag_row_indices, non_diag_col_indices), dim=0)

    diag = torch.arange(0, size, device=device)
    diag_row_indices = (row_template + total_d * diag.unsqueeze(1)).reshape(1, -1)
    diag_col_indices = (col_template + total_d * diag.unsqueeze(1)).reshape(1, -1)
    diag_indices = torch.cat((diag_row_indices, diag_col_indices), dim=0)

    return diag_indices, non_diag_indices


# 合并两个没有重复条目的稀疏矩阵索引和值。
def mergesp(index1, value1, index2, value2):
    assert index1.dim() == 2 and index2.dim() == 2
    assert value1.dim() == 1 and value2.dim() == 1
    assert index1.size(1) == value1.numel()
    assert index2.size(1) == value2.numel()
    assert index1.size(0) == 2 and index2.size(0) == 2

    index = torch.cat([index1, index2], dim=1)
    val = torch.cat([value1, value2])
    return index, val
