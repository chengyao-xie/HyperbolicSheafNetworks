from utils.brain_dataset import (
    preprocess_dataset,
)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description='Preprocess brain graph datasets with 2-simplicial complexes'
    )
    parser.add_argument('--dataset', type=str,
                        choices=['abide', 'ppmi', 'taowu', 'neurocon', 'Matai', 'all'],
                        default='abide', help='Dataset to preprocess')
    parser.add_argument('--top_k_percent', type=int, default=20,
                        help='Keep top k strongest edges')
    parser.add_argument('--max_triangles', type=int, default=500,
                        help='Maximum triangles per graph')
    parser.add_argument('--simplex_order', type=int, default=2, choices=[1, 2],
                        help='Order of simplicial complex: 1 (nodes+edges) or 2 (nodes+edges+triangles)')
    parser.add_argument('--construction_method', type=str, default='pearson',
                        choices=['pearson', 'causality'], help='Graph construction method')
    parser.add_argument('--cs_embedding_dim', type=int, default=3,
                        help='Takens embedding dimension for continuity scaling')
    parser.add_argument('--cs_delay', type=int, default=1,
                        help='Takens embedding delay for continuity scaling')
    parser.add_argument('--cs_scales', type=int, default=33,
                        help='Number of scales for continuity scaling')
    parser.add_argument('--cs_theiler_window', type=int, default=2,
                        help='Temporal exclusion window for continuity scaling')
    parser.add_argument('--cs_epsilon_min', type=float, default=0.001,
                        help='Minimum epsilon percent for continuity scaling')
    parser.add_argument('--cs_epsilon_max', type=float, default=1.0,
                        help='Maximum epsilon percent for continuity scaling')
    return parser.parse_args()


def main():
    args = parse_args()
    max_triangles = args.max_triangles if args.max_triangles > 0 else None
    print(
        f"Configuration: top_k={args.top_k_percent}%, "
        f"max_triangles={max_triangles if max_triangles else 'unlimited'}, "
        f"simplex_order={args.simplex_order}, "
        f"construction={args.construction_method}"
    )

    datasets = ['abide', 'ppmi', 'taowu', 'neurocon', 'Matai'] if args.dataset == 'all' else [args.dataset]
    success = 0
    for dataset_name in datasets:
        print(f"\nProcessing {dataset_name.upper()} - {args.construction_method}...")
        result = preprocess_dataset(
            dataset_name,
            args.top_k_percent,
            max_triangles=max_triangles,
            simplex_order=args.simplex_order,
            construction_method=args.construction_method,
            cs_embedding_dim=args.cs_embedding_dim,
            cs_delay=args.cs_delay,
            cs_scales=args.cs_scales,
            cs_theiler_window=args.cs_theiler_window,
            cs_epsilon_min=args.cs_epsilon_min,
            cs_epsilon_max=args.cs_epsilon_max,
        )
        if result:
            success += 1

    print(f"Completed: {success}/{len(datasets)} successful")


if __name__ == "__main__":
    main()
