import json
import logging
import os
import sys
from copy import copy
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from exp.training import run_fold, set_seed
from utils.brain_dataset import get_dataset


class PrintLogger:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip():
            self.logger.info(message.rstrip())

    def flush(self):
        pass


def create_run_dir(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(
        os.path.dirname(__file__),
        '..',
        'log',
        f'run_{args.dataset}_seed{args.seed}_repeats{args.num_repeats}_{timestamp}',
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def configure_logging(run_dir):
    log_path = os.path.join(run_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger()


def print_run_config(args, dataset):
    print(f"Graphs={len(dataset)} | Features={dataset.num_features} | Classes={dataset.num_classes}")
    print(f"{args.dataset} | order={args.simplex_order} | {args.construction_method} | "
          f"seed={args.seed} | repeats={args.num_repeats}")
    print(f"layers={args.layers} | hidden={args.hidden_channels} | d={args.d} | "
          f"pool={args.pool_type} | max_t={args.max_t}")
    print(f"AdamW lr={args.lr} | decay={args.weight_decay} | epochs={args.epochs} | "
          f"early_stop={args.early_stopping} | checkpoint=loss")


def print_label_distribution(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples ({count / len(labels) * 100:.1f}%)")


def summarize_results(repeat_summaries):
    print("Final Results (mean +/- std over repeated five-fold means)")
    labels = {
        'accuracy': 'Test Accuracy',
        'balanced_accuracy': 'Test Balanced Acc',
        'sensitivity': 'Test Sensitivity',
        'specificity': 'Test Specificity',
        'auc': 'Test AUC',
    }
    for name, label in labels.items():
        values = [summary[name] for summary in repeat_summaries]
        print(f"{label:<21} {np.mean(values) * 100:.2f} +/- {np.std(values) * 100:.2f}%")


def serialise_arguments(args):
    return {
        name: (str(value) if isinstance(value, torch.device) else value)
        for name, value in vars(args).items()
    }


def save_fold_metrics(run_dir, args, folds, repeat_summaries):
    metrics = {
        'protocol': 'formal',
        'dataset': args.dataset,
        'construction_method': args.construction_method,
        'simplex_order': args.simplex_order,
        'base_seed': args.seed,
        'num_repeats': args.num_repeats,
        'hyperparameters': serialise_arguments(args),
        'folds': folds,
    }
    metric_names = ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'auc')
    metrics['repeat_summaries'] = repeat_summaries
    metrics['summary'] = {
        name: {
            'mean': float(np.mean([summary[name] for summary in repeat_summaries])),
            'std': float(np.std([summary[name] for summary in repeat_summaries])),
        }
        for name in metric_names
    } if repeat_summaries else {}
    with open(os.path.join(run_dir, 'fold_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    run_dir = create_run_dir(args)
    logger = configure_logging(run_dir)

    original_stdout = sys.stdout
    sys.stdout = PrintLogger(logger)

    try:
        dataset = get_dataset(
            args.dataset,
            args.simplex_order,
            construction_method=args.construction_method,
            preprocess_workers=args.preprocess_workers,
        )
        print_run_config(args, dataset)

        all_labels = np.array([data.y.item() for data in dataset])
        print_label_distribution(all_labels)

        fold_metric_records = []
        repeat_summaries = []
        print(f"Folds: {args.num_folds}")
        repeat_splits = []
        for repeat_idx in range(args.num_repeats):
            repeat_seed = args.seed + repeat_idx
            skf = StratifiedKFold(
                n_splits=args.num_folds,
                shuffle=True,
                random_state=repeat_seed,
            )
            fold_splits = list(skf.split(np.zeros(len(all_labels)), all_labels))
            repeat_splits.append((repeat_idx, repeat_seed, fold_splits))
        metric_names = ('accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'auc')
        for repeat_idx, repeat_seed, fold_splits in repeat_splits:
            print(f"Repeat {repeat_idx + 1}/{args.num_repeats} | seed={repeat_seed}")
            set_seed(repeat_seed)
            repeat_args = copy(args)
            repeat_args.seed = repeat_seed
            repeat_values = {name: [] for name in metric_names}
            for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
                test_acc, test_metrics, fold_selection = run_fold(
                    repeat_args, dataset, fold_idx, train_idx, test_idx, run_dir)
                del test_acc
                for name in metric_names:
                    repeat_values[name].append(float(test_metrics[name]))
                fold_metric_records.append({
                    'repeat': repeat_idx + 1,
                    'seed': repeat_seed,
                    'fold': fold_idx + 1,
                    'metrics': {name: float(test_metrics[name]) for name in metric_names},
                    'selection': fold_selection,
                })
            repeat_summaries.append({
                name: float(np.mean(values)) for name, values in repeat_values.items()
            })
            save_fold_metrics(run_dir, args, fold_metric_records, repeat_summaries)

        summarize_results(repeat_summaries)
    finally:
        sys.stdout = original_stdout


if __name__ == '__main__':
    main()
