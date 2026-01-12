# 分析 .pt脚本文件

import os
import argparse
import torch
from pprint import pprint
from torch_geometric.data import Data

def tensor_to_len(t):
    try:
        return int(t.size(0))
    except Exception:
        try:
            return int(len(t))
        except Exception:
            return None

def get_num_graphs_from_slices(slices):
    # slices 的任意 key 对应的长度是 num_graphs+1
    for v in slices.values():
        n = tensor_to_len(v)
        if n and n > 0:
            return n - 1
    return 0

def reconstruct_graph_from_collated(data, slices, idx):
    # data: Data-like with attributes as tensors concatenated
    # slices: dict of same keys -> cumulative indices
    from torch_geometric.data import Data as TGData
    graph = TGData()
    node_start = 0
    if 'x' in slices:
        node_start = int(slices['x'][idx].item()) if hasattr(slices['x'], 'item') else int(slices['x'][idx])
    # copy attributes that have slices
    for key, slice_arr in slices.items():
        start = int(slice_arr[idx].item()) if hasattr(slice_arr[idx], 'item') else int(slice_arr[idx])
        end = int(slice_arr[idx+1].item()) if hasattr(slice_arr[idx+1], 'item') else int(slice_arr[idx+1])
        val = getattr(data, key, None)
        if val is None:
            continue
        # edge_index needs column slicing and index shift
        if key == 'edge_index':
            chunk = val[:, start:end].clone()
            # shift node indices to be local
            if node_start != 0:
                chunk = chunk - node_start
            graph.edge_index = chunk
        else:
            # typical case: rows correspond to nodes or graphs
            try:
                graph.__setattr__(key, val[start:end].clone())
            except Exception:
                # fallback: assign as-is (for scalars)
                graph.__setattr__(key, val[start:end])
    # also try to attach global attributes present in data but not in slices
    for key in data.keys:
        if key in slices:
            continue
        if hasattr(data, key):
            setattr(graph, key, getattr(data, key))
    return graph

def inspect_pt(path, show_sample=None, max_print=5):
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return
    obj = torch.load(path, weights_only=False)
    print(f"Loaded: {path}  (type={type(obj)})")
    # Case A: list of Data objects
    if isinstance(obj, list):
        print("Format: list of Data objects")
        print(f"Number of samples: {len(obj)}")
        total_nodes = sum([getattr(d, 'x').size(0) if getattr(d, 'x', None) is not None else 0 for d in obj])
        total_edges = sum([getattr(d, 'edge_index').size(1) if getattr(d, 'edge_index', None) is not None else 0 for d in obj])
        print(f"Total nodes: {total_nodes}, total edges: {total_edges}")
        print("Sample keys:", obj[0].keys if len(obj) > 0 else [])
        for i, d in enumerate(obj[:max_print]):
            print(f"--- Sample {i} ---")
            print("num_nodes:", getattr(d, 'x').size(0) if getattr(d, 'x', None) is not None else None)
            print("num_edges:", getattr(d, 'edge_index').size(1) if getattr(d, 'edge_index', None) is not None else None)
            if hasattr(d, 'y'):
                print("y:", d.y)
        if show_sample is not None:
            idx = show_sample
            if idx < 0 or idx >= len(obj):
                print("sample index out of range")
            else:
                print(f"Full sample {idx}:")
                pprint(dict(obj[idx]))
        return

    # Case B: tuple (data, slices) from InMemoryDataset.collate
    if isinstance(obj, (tuple, list)) and len(obj) == 2:
        data, slices = obj
        print("Format: (data, slices) from InMemoryDataset")
        print("Data keys:", data.keys)
        print("Slices keys:", list(slices.keys()))
        num_graphs = get_num_graphs_from_slices(slices)
        print(f"Number of graphs (from slices): {num_graphs}")
        total_nodes = data.x.size(0) if hasattr(data, 'x') else None
        total_edges = data.edge_index.size(1) if hasattr(data, 'edge_index') else None
        print(f"Total nodes (concatenated): {total_nodes}, total edges (concatenated): {total_edges}")
        # show per-sample basic stats for first few
        for i in range(min(num_graphs, max_print)):
            nstart = int(slices['x'][i].item()) if 'x' in slices else None
            nend = int(slices['x'][i+1].item()) if 'x' in slices else None
            estart = int(slices['edge_index'][i].item()) if 'edge_index' in slices else None
            eend = int(slices['edge_index'][i+1].item()) if 'edge_index' in slices else None
            num_nodes = (nend - nstart) if (nstart is not None and nend is not None) else None
            num_edges = (eend - estart) if (estart is not None and eend is not None) else None
            print(f"Sample {i}: nodes={num_nodes}, edges={num_edges}")
            # try label
            if 'y' in slices:
                y_start = int(slices['y'][i].item())
                y_end = int(slices['y'][i+1].item())
                try:
                    yval = data.y[y_start:y_end]
                except Exception:
                    yval = None
                print("  y:", yval)
        if show_sample is not None:
            if show_sample < 0 or show_sample >= num_graphs:
                print("sample index out of range")
            else:
                g = reconstruct_graph_from_collated(data, slices, show_sample)
                print(f"Full reconstructed sample {show_sample}:")
                pprint(dict(g))
        return

    # Case C: single Data object
    if isinstance(obj, Data):
        d = obj
        print("Format: single torch_geometric.data.Data")
        print("Keys:", d.keys)
        print("num_nodes:", getattr(d, 'x').size(0) if getattr(d, 'x', None) is not None else None)
        print("num_edges:", getattr(d, 'edge_index').size(1) if getattr(d, 'edge_index', None) is not None else None)
        if hasattr(d, 'y'):
            print("y:", d.y)
        if show_sample is not None:
            print("single Data object has no sample index")
        return

    # fallback
    print("Unrecognized file content. Type:", type(obj))
    pprint(obj)

def main():
    this_dir = os.path.dirname(__file__)
    default_pt = os.path.normpath(os.path.join(this_dir, "abide_node_labeled_dataset.pt"))
    parser = argparse.ArgumentParser(description="Inspect a node_labeled_dataset.pt file")
    parser.add_argument("--path", "-p", default=default_pt, help="path to .pt file")
    parser.add_argument("--show-sample", "-s", type=int, default=None, help="show reconstructed sample index")
    parser.add_argument("--max-print", type=int, default=5, help="how many samples to preview")
    args = parser.parse_args()
    inspect_pt(args.path, show_sample=args.show_sample, max_print=args.max_print)

if __name__ == "__main__":
    main()
