import torch
import numpy as np


"""
Constructs a 2-skeleton (nodes, edges, triangles) from a binary adjacency matrix.

Parameters
----------
adj : (n,n) numpy array
    Binary adjacency matrix (symmetric, 0/1 or logical)

Returns
-------
pSkeleton : list of arrays
    pSkeleton[0] : (n,1) node indices
    pSkeleton[1] : (m,2) edge list (i<j)
    pSkeleton[2] : (k,3) triangle list (sorted indices)
"""
def hodge_2skeleton(adj):
    n = adj.shape[0]

    nlist = np.arange(1, n + 1)[:, np.newaxis]

    adj1 = np.abs(adj)
    i, j = np.triu_indices(n, k=1)
    edge_mask = adj1[i, j] > 0

    elist = np.column_stack((i[edge_mask] + 1, j[edge_mask] + 1))
    elist = elist[np.lexsort((elist[:, 1], elist[:, 0]))]

    adjnew = (adj1 > 0).astype(int)
    lower_adj = np.tril(adjnew)

    triangles = []
    for edge_idx in range(elist.shape[0]):
        u, v = elist[edge_idx]
        u_idx, v_idx = u - 1, v - 1

        common = (lower_adj[:, u_idx] == 1) & (lower_adj[:, v_idx] == 1)
        candidates = np.where(common)[0] + 1
        candidates = candidates[candidates > v]

        for w in candidates:
            triangles.append(sorted([u, v, w]))

    if triangles:
        tlist = np.unique(np.array(triangles), axis=0)
    else:
        tlist = np.empty((0, 3), dtype=int)

    return [nlist, elist, tlist]


"""
Build oriented boundary matrix sigma2 (edges x triangles).
Expects 1-based indexing in edges and triangles.
"""
def triangle_to_edge_matrix(triangles, edges):
    edges = [tuple(e) for e in edges]
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}

    sigma2 = torch.zeros((len(edges), len(triangles)), dtype=torch.float)

    for col, tri in enumerate(triangles):
        i, j, k = tri
        oriented_edges = [
            ((i, j), (j, i), 1),
            ((j, k), (k, j), 1),
            ((i, k), (k, i), -1),
        ]

        for forward, backward, sign in oriented_edges:
            if forward in edge_to_idx:
                sigma2[edge_to_idx[forward], col] += sign
            elif backward in edge_to_idx:
                sigma2[edge_to_idx[backward], col] -= sign

    return sigma2


# Build incidence matrix sigma1 (nodes x edges).
def edge_to_node_matrix(edges, nodes, one_indexed=True):
    sigma1 = torch.zeros((len(nodes), len(edges)), dtype=torch.float)
    offset = int(one_indexed)

    for col, edge in enumerate(edges):
        x, y = edge
        sigma1[x - offset][col] -= 1
        sigma1[y - offset][col] += 1

    return sigma1


# Convert a boundary matrix into column-row incidence pairs and signs.
def boundary_to_incidence_data(boundary):
    if boundary.layout == torch.strided:
        coordinates = torch.nonzero(boundary, as_tuple=False)
        if coordinates.numel() == 0:
            index = torch.empty((2, 0), dtype=torch.long, device=boundary.device)
            signs = torch.empty((0,), dtype=boundary.dtype, device=boundary.device)
            return index, signs
        lower_index, upper_index = coordinates.unbind(dim=1)
        signs = boundary[lower_index, upper_index]
    else:
        boundary = boundary.to_sparse_coo().coalesce()
        lower_index, upper_index = boundary.indices()
        signs = boundary.values()

    return torch.stack((upper_index, lower_index), dim=0).long(), signs
