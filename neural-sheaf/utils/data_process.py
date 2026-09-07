import argparse

from utils.brain_dataset import preprocess_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Build formal-experiment dataset caches')
    parser.add_argument(
        '--dataset',
        choices=['abide', 'ppmi', 'taowu', 'neurocon', 'Matai', 'all'],
        default='abide',
    )
    parser.add_argument(
        '--construction_method', choices=['causality', 'pearson'],
        default='causality')
    parser.add_argument('--simplex_order', type=int, choices=[1, 2], default=2)
    parser.add_argument('--workers', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = (
        ['abide', 'ppmi', 'taowu', 'neurocon', 'Matai']
        if args.dataset == 'all' else [args.dataset]
    )
    print(f"Preprocessing {len(datasets)} dataset(s) with the formal configuration.")

    completed = 0
    for dataset_name in datasets:
        print(f"Processing {dataset_name.upper()}...")
        if preprocess_dataset(
                dataset_name,
                construction_method=args.construction_method,
                simplex_order=args.simplex_order,
                preprocess_workers=args.workers,
        ):
            completed += 1

    print(f"Completed: {completed}/{len(datasets)}")


if __name__ == "__main__":
    main()
