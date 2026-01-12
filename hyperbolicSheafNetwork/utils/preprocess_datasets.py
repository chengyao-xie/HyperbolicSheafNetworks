import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch_geometric.data import InMemoryDataset
from utils.GraphDataset import brain_graph_dataset


def preprocess_all_datasets(dataset='all', graph_type='both', order='both'):
    this_dir = os.path.dirname(__file__)
    
    dataset_paths = {
        'abide': "/home/featurize/data/abide",
        'ppmi': "/home/featurize/data/ppmi",
        'taowu': "/home/featurize/data/taowu",
        'neurocon': "/home/featurize/data/neurocon",
        'Matai': "/home/featurize/data/Matai",
    }

    target_datasets = dataset_paths.items() if dataset == 'all' else [(dataset, dataset_paths.get(dataset))]
    target_graph_types = ['correlation', 'causality'] if graph_type == 'both' else [graph_type]
    target_orders = [1, 2] if order == 'both' else [int(order)]
    
    for dataset_name, root_dir in target_datasets:
        if root_dir is None:
            print(f"Warning: Unknown dataset {dataset_name}, skipping.")
            continue

        for graph_type in target_graph_types:
            for ord_level in target_orders:
                print(f"\n{'='*60}")
                print(f"Processing dataset: {dataset_name} ({graph_type}) order={ord_level}")
                print(f"{'='*60}")
                
                pt_path = os.path.join(
                    os.path.dirname(this_dir),
                    'dataset',
                    f"{dataset_name}_{graph_type}_order{ord_level}_graph_labeled_dataset.pt"
                )

                if not os.path.isdir(root_dir):
                    print(f"Warning: Raw data directory not found: {root_dir}. Skipping {dataset_name}.")
                    continue

                data_list = brain_graph_dataset(root_dir, top_k_percent=80, graph_type=graph_type, order=ord_level)
                
                if len(data_list) == 0:
                    print(f"Warning: No samples built from raw files for {dataset_name} ({graph_type}) order={ord_level}. Skipping.")
                    continue

                data, slices = InMemoryDataset.collate(data_list)
                os.makedirs(os.path.dirname(pt_path), exist_ok=True)
                torch.save((data, slices), pt_path)
                print(f"Saved dataset to {pt_path} (num_samples={len(data_list)})")
    
    print(f"\n{'='*60}")
    print("Dataset preprocessing completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess brain graph datasets from raw .mat files')
    parser.add_argument('--dataset', type=str, 
                        choices=['abide', 'ppmi', 'taowu', 'neurocon', 'Matai', 'all'],
                        default='all',
                        help='Dataset to preprocess (default: all)')
    parser.add_argument('--graph_type', type=str,
                        choices=['correlation', 'causality', 'both'],
                        default='both',
                        help='Graph type to preprocess (default: both)')
    parser.add_argument('--order', type=str,
                        choices=['1', '2', 'both'],
                        default='1',
                        help='Network order to preprocess (1 for first-order, 2 for second-order, both for both)')
    
    args = parser.parse_args()
    
    preprocess_all_datasets(dataset=args.dataset, graph_type=args.graph_type, order=args.order)
