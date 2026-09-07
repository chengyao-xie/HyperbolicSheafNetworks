import torch



# compute the left/right map index for each undirected edge in a bidirectional edge
def compute_left_right_map_index(edge_index, full_matrix=False):
    """Match every directed edge with its reverse without host-device syncs."""
    if edge_index.numel() == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return empty, empty

    row, col = edge_index
    num_nodes = edge_index.max() + 1
    edge_keys = row * num_nodes + col
    reverse_keys = col * num_nodes + row
    sorted_keys, permutation = torch.sort(edge_keys)
    reverse_positions = torch.searchsorted(sorted_keys, reverse_keys)

    reverse_index = permutation.index_select(0, reverse_positions)
    if full_matrix:
        left_index = torch.arange(edge_index.size(1), device=edge_index.device)
    else:
        left_index = torch.nonzero(row < col, as_tuple=False).view(-1)
    right_index = reverse_index.index_select(0, left_index)
    left_right_index = torch.stack([left_index, right_index], dim=0)
    return left_right_index, edge_index.index_select(1, left_index)



# calculate learnable part sparse indices in diagonal sheaf Laplacian
def compute_learnable_diag_laplacian_indices(size, edge_index, learned_d, total_d):
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


# calculate fixed low/high frequency indices in diagonal Laplacian
def compute_fixed_diag_laplacian_indices(size, edge_index, learned_d, total_d):
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


# merge the two sparse matrix indices and values without duplicates
def mergesp(index1, value1, index2, value2):
    index = torch.cat([index1, index2], dim=1)
    val = torch.cat([value1, value2])
    return index, val
