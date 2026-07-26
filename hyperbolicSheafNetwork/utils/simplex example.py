import os
import torch
import numpy as np
import scipy.io
from torch_geometric.data import Data, InMemoryDataset

# https://github.com/laplcebeltrami/hodge
def hodge_2skeleton(adj):
    """
    Constructs a 2-skeleton (nodes, edges, triangles) from a binary adjacency matrix.
    
    This is a Python translation of the MATLAB function by Anand & Chung (2021)
    "Hodge-Laplacian of brain networks and its application to modeling cycles"
    
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
    
    n = adj.shape[0]
    
    # 0-skeleton: just the node list
    nlist = np.arange(1, n+1)[:, np.newaxis]   # shape (n,1)
    
    # 1-skeleton: edge list
    adj1 = np.abs(adj)              # in case someone passed signed values
    # Get upper triangular indices (i < j)
    i, j = np.triu_indices(n, k=1)
    edge_mask = adj1[i, j] > 0
    
    elist = np.column_stack((i[edge_mask]+1, j[edge_mask]+1))  # +1 for 1-based indexing
    elist = elist[np.lexsort((elist[:,1], elist[:,0]))]        # sort like sortrows
    
    # 2-skeleton: triangles
    # Convert to binary (you can adjust threshold logic here)
    threshold = 0
    adjnew = (adj1 > threshold).astype(int)
    
    # We only need lower triangle for the column check
    G = np.tril(adjnew)            # G[i,j] = 1 only if i > j and connected
    
    triangles = []
    
    for edge_idx in range(elist.shape[0]):
        u, v = elist[edge_idx]      # 1-based
        u_idx, v_idx = u-1, v-1     # 0-based for indexing
        
        # Find nodes w > v such that w-v and w-u exist
        # i.e. look at column v and column u in lower triangle part
        col_u = G[:, u_idx]
        col_v = G[:, v_idx]
        
        # both edges must exist → logical AND
        common = (col_u == 1) & (col_v == 1)
        
        candidates = np.where(common)[0] + 1   # back to 1-based
        
        # only keep w > v (to avoid duplicates and keep orientation)
        candidates = candidates[candidates > v]
        
        for w in candidates:
            triangle = sorted([u, v, w])
            triangles.append(triangle)
    
    # Remove duplicate triangles (if any)
    if triangles:
        tlist = np.array(triangles)
        tlist = np.unique(tlist, axis=0)
    else:
        tlist = np.empty((0, 3), dtype=int)
    
    # Pack everything together (MATLAB cell-like)
    pSkeleton = [
        nlist,          # 0-skeleton: nodes
        elist,          # 1-skeleton: edges
        tlist           # 2-skeleton: triangles
    ]
    
    return pSkeleton




# https://github.com/ggoh29/Simplicial-neural-network-benchmark/blob/main/utils.py
def triangle_to_edge_matrix(triangles, edges):
    """
    Build oriented boundary matrix σ₂ (edges × triangles)
    Expects 1-based indexing in edges and triangles
    """
    # Convert list of lists → list of tuples (so they become hashable)
    edges = [tuple(e) for e in edges]           # crucial fix

    # Now we can safely create the lookup dictionary
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}

    sigma2 = torch.zeros((len(edges), len(triangles)), dtype=torch.float)

    for l, tri in enumerate(triangles):
        i, j, k = tri   # assume already sorted or not — doesn't matter for this logic

        # Edge i-j
        e_ij = (i, j)
        e_ji = (j, i)
        if e_ij in edge_to_idx:
            sigma2[edge_to_idx[e_ij], l] += 1
        elif e_ji in edge_to_idx:
            sigma2[edge_to_idx[e_ji], l] -= 1

        # Edge j-k
        e_jk = (j, k)
        e_kj = (k, j)
        if e_jk in edge_to_idx:
            sigma2[edge_to_idx[e_jk], l] += 1
        elif e_kj in edge_to_idx:
            sigma2[edge_to_idx[e_kj], l] -= 1

        # Edge i-k   (note: sign is usually opposite in standard boundary operator)
        e_ik = (i, k)
        e_ki = (k, i)
        if e_ik in edge_to_idx:
            sigma2[edge_to_idx[e_ik], l] -= 1
        elif e_ki in edge_to_idx:
            sigma2[edge_to_idx[e_ki], l] += 1

    return sigma2

# one_indexed: whether the input edges are start at 1 (True) or 0 (False)
def edge_to_node_matrix(edges, nodes, one_indexed=True):
    sigma1 = torch.zeros((len(nodes), len(edges)), dtype=torch.float)
    offset = int(one_indexed)
    j = 0
    for edge in edges:
        x, y = edge
        sigma1[x - offset][j] -= 1
        sigma1[y - offset][j] += 1
        j += 1
    return sigma1


# ────────────────────────────────────────────────
#   Quick test / usage example
# ────────────────────────────────────────────────
if __name__ == "__main__":
    # small example - cycle of 4 + one diagonal
    # https://www.mdpi.com/2227-7390/13/2/208#mathematics-13-00208-f006
    #    1 ----- 2
    #    | \     |
    #    |   \   |
    #    |     \ |
    #    4 ----- 3
    #
    A = np.array([
        [0,1,1,1],
        [1,0,1,0],
        [1,1,0,1],
        [1,0,1,0]
    ])

    skel = hodge_2skeleton(A)

    print("Nodes:")
    print(skel[0])
    print("\nEdges:")
    print(skel[1])
    print("\nTriangles:")
    print(skel[2])


    print("--------------------------------------------------  ")
    # ─── Example simplicial complex ─────────────────────────────────────

    nodes = skel[0].flatten().tolist()
    edges = skel[1].tolist()
    # # 手动将边的顺序调整为论文的顺序
    # edges = [[1,2],[2,3],[3,4],[1,4],[1,3]]
    triangles =  skel[2].tolist()

    # B_1: nodes -> edges, 关联矩阵 incidence matrix
    # 在我们的工作里，1阶数据上，用的是可学习的sheaf来取代固定的关联矩阵 incidence matrix
    sigma1 = edge_to_node_matrix(edges, nodes)
    print("\nBoundary matrix σ₁ = ∂₁ (shape:", sigma1.shape, ")")
    print(sigma1)

    # B_2: edges -> triangles
    sigma2 = triangle_to_edge_matrix(triangles, edges)

    print("Edges (row indices):")
    for i, e in enumerate(edges):
        print(f"{i:2d} : {e}")

    print("\nTriangles (columns):")
    for j, t in enumerate(triangles):
        print(f"{j:2d} : {t}")

    print("\nBoundary matrix σ₂ = ∂₂ (shape:", sigma2.shape, ")")
    print(sigma2)


    print("--------------------------------------------------  ")
    # ─── Classical graph Laplacian L = D − A ────────────────────────
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L_classic = D - A

    print("\nGraph Laplacian L = D − A:")
    print(L_classic.round(1))   # for readability

    # ─── Incidence matrix B₁  (nodes → edges) ─────────────────
    # https://en.wikipedia.org/wiki/Laplacian_matrix
    B1 = sigma1
    print("\nIncidence matrix B₁ (shape:", B1.shape, ")")
    print(B1)

    # ─── Reconstruct Laplacian: B₁ B₁ᵀ ──────────────────────────────
    L_from_B = B1 @ B1.t()

    print("\nReconstructed Laplacian B₁ B₁ᵀ:")
    print(L_from_B.numpy().round(1))

    # ─── Verification ────────────────────────────────────────────────
    diff = np.abs(L_classic - L_from_B.numpy())
    max_diff = diff.max()
    is_equal = np.allclose(L_classic, L_from_B.numpy(), atol=1e-10)

    print("\nVerification L == B₁ B₁ᵀ ?")
    print(f"  → Maximum absolute difference: {max_diff:.2e}")
    print(f"  → Matrices are equal (within tolerance): {is_equal}")

    if is_equal:
        print("\nSUCCESS: L = D − A = B₁ B₁ᵀ holds ✓")
    else:
        print("\nMismatch detected — check indexing or orientation convention.")


    print("--------------------------------------------------  ")
    # ─── 1-Hodge Laplacian / Helmholtzian on edges ───────────────
    print("\n∂₂ = sigma2  (edges × triangles) shape:", sigma2.shape)
    print(sigma2.numpy().round(1))

    # Δ₁ = ∂₂ᵀ ∂₂ + ∂₁ ∂₁ᵀ     shape (m × m)
    delta1 = sigma2 @ sigma2.t() + sigma1.t() @ sigma1

    print("\nHelmholtzian / 1-Hodge Laplacian on 1-chains Δ₁:")
    print("Δ₁ = ∂₂ᵀ ∂₂ + ∂₁ ∂₁ᵀ   shape:", delta1.shape)
    print(delta1.numpy().round(2))

    # ─── Interpretation / diagonal elements ──────────────────────
    print("\nDiagonal of Δ₁ (local contributions):")
    print(np.diag(delta1.numpy()).round(2))

    print("\nNote:")
    print("• Diagonal entries = degree(e) + number of triangles containing e")
    print("• Off-diagonal = -1 if two edges share a triangle (coface relation)")
    print("  (with sign depending on orientation consistency)")
    print("--------------------------------------------------  ")

    # # https://www.math.uni-bielefeld.de/~grigor/hodge1.pdf   page 14
    # # example from the article
    # A2 = np.array([
    #     [1,1,1],
    #     [1,1,1],
    #     [1,1,1]
    # ])
    # skel2 = hodge_2skeleton(A2)
    # # # 手动将边的顺序调整为ppt的顺序
    # # edges2 = [[1,2],[2,3],[1,3]]
    # # edges2 = [[1,2],[2,3],[3,1]]
    # edges2 = skel2[1].tolist()
    # triangles2 =  skel2[2].tolist()
    # sigma_2 = triangle_to_edge_matrix(triangles2, edges2)
    # print("\nBoundary matrix σ₂ = ∂₂ (shape:", sigma_2.shape, ")")
    # print(sigma_2)
    # sigma_1 = edge_to_node_matrix(edges2, skel2[0].flatten().tolist())
    # print("\nBoundary matrix σ₁ = ∂₁ (shape:", sigma_1.shape, ")")
    # print(sigma_1)  
    # delta_1 = sigma_1 @ sigma_1.t()
    # print("\n graph Laplacian on nodes Δ₀ = ∂₁ ∂₁ᵀ:")
    # print("Δ₀ = ∂₁ ∂₁ᵀ   shape:", delta_1.shape)
    # print(delta_1.numpy().round(2))
          
    # delta_2 = sigma_2 @ sigma_2.t() + sigma_1.t() @ sigma_1
    # print("\nHelmholtzian / 1-Hodge Laplacian on 1-chains Δ₁:")
    # print("Δ₁ = ∂₂ᵀ ∂₂ + ∂₁ ∂₁ᵀ   shape:", delta_2.shape)
    # print(delta_2.numpy().round(2)) 


    # print("--------------------------------------------------  ")
    # A3 = np.array([
    #     [0,1,1,0],
    #     [1,0,0,1],
    #     [1,0,0,1],
    #     [0,1,1,0]
    # ])
    # skel3 = hodge_2skeleton(A3)
    # print("\nSimplicial complex from complete graph K4:")
    # nodes = skel3[0].flatten().tolist()
    # edges = skel3[1].tolist()
    # triangles =  skel3[2].tolist()
    # sigma3_2 = triangle_to_edge_matrix(triangles, edges)
    # print("\nBoundary matrix σ₂ = ∂₂ (shape:", sigma3_2.shape, ")")
    # sigma3_1 = edge_to_node_matrix(edges, nodes)
    # print("\nBoundary matrix σ₁ = ∂₁ (shape:", sigma3_1.shape, ")")
    # delta3_1 = sigma3_2 @ sigma3_2.t() + sigma3_1.t() @ sigma3_1
    # print("\nHelmholtzian / 1-Hodge Laplacian on 1-chains Δ₁:")
    # print("Δ₁ = ∂₂ᵀ ∂₂ + ∂₁ ∂₁ᵀ   shape:", delta3_1.shape)
    # print(delta3_1.numpy().round(2))