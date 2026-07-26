import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from exp.training import run_fold, set_seed
from utils.brain_dataset import get_dataset
from utils.resource_monitor import ResourceTracker, format_resource_summary, save_resource_report


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
        f'run_{args.dataset}_seed{args.seed}_{timestamp}',
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
    print(f"Loaded {len(dataset)} graphs | Features: {dataset.num_features} | Classes: {dataset.num_classes}")
    print(f"Dataset: {args.dataset} | Simplex Order: {args.simplex_order} | Seed: {args.seed}")
    print(f"Construction: {args.construction_method} | Batch Size: {args.batch_size} | Optimizer: {args.optimizer}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Weight Decay: {args.weight_decay}")
    print(f"Hidden Channels: {args.hidden_channels} | Dropout: {args.dropout}/{args.input_dropout}")
    print(f"Pool: {args.pool_type} | Early Stop: {args.early_stopping} | Grad Clip: {args.grad_clip}")
    print(f"AMP: {args.amp} | Compile: {args.compile_model} | Scheduler: {args.scheduler}")
    print(f"Eval Interval: {args.eval_interval} | Log Interval: {args.log_interval} | "
          f"Workers: {args.num_workers} | Accumulation: {args.accumulation_steps}")
    print(f"Best Metric: {args.checkpoint_metric} | Save Model: {args.save_model}")


def print_label_distribution(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  Class {label}: {count} samples ({count / len(labels) * 100:.1f}%)")


def summarize_results(test_accs, test_bals, test_sens, test_spes, test_aucs):
    print("Final Results (Averaged over Repeats)")
    print(f"Test Accuracy:       {np.mean(test_accs) * 100:.2f} +/- {np.std(test_accs) * 100:.2f}%")
    print(f"Test Balanced Acc:   {np.mean(test_bals) * 100:.2f} +/- {np.std(test_bals) * 100:.2f}%")
    print(f"Test Sensitivity:    {np.mean(test_sens) * 100:.2f} +/- {np.std(test_sens) * 100:.2f}%")
    print(f"Test Specificity:    {np.mean(test_spes) * 100:.2f} +/- {np.std(test_spes) * 100:.2f}%")
    print(f"Test AUC:            {np.mean(test_aucs) * 100:.2f} +/- {np.std(test_aucs) * 100:.2f}%")


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay
    return args


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic, benchmark=args.cudnn_benchmark)

    run_dir = create_run_dir(args)
    logger = configure_logging(run_dir)

    original_stdout = sys.stdout
    sys.stdout = PrintLogger(logger)

    try:
        total_tracker = ResourceTracker('total_run', args.device).start()
        dataset = get_dataset(
            args.dataset,
            args.simplex_order,
            top_k_percent=args.top_k_percent,
            max_triangles=args.max_triangles if args.max_triangles > 0 else None,
            construction_method=args.construction_method,
            cs_embedding_dim=args.cs_embedding_dim,
            cs_delay=args.cs_delay,
            cs_scales=args.cs_scales,
            cs_theiler_window=args.cs_theiler_window,
            cs_epsilon_min=args.cs_epsilon_min,
            cs_epsilon_max=args.cs_epsilon_max,
            preprocess_workers=args.preprocess_workers,
        )
        print_run_config(args, dataset)

        all_labels = np.array([data.y.item() for data in dataset])
        print_label_distribution(all_labels)

        test_accs, test_bals, test_sens, test_spes, test_aucs = [], [], [], [], []
        fold_resource_records = []
        fold_limit = args.num_folds if args.fold_limit <= 0 else min(args.fold_limit, args.num_folds)
        print(f"Fold Limit: {fold_limit}/{args.num_folds}")
        skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(all_labels)), all_labels)):
            if fold_idx >= fold_limit:
                break
            test_acc, test_metrics, fold_resources = run_fold(args, dataset, fold_idx, train_idx, test_idx, run_dir)
            test_accs.append(test_acc)
            test_bals.append(test_metrics.get('balanced_accuracy', 0.0))
            test_sens.append(test_metrics.get('sensitivity', 0.0))
            test_spes.append(test_metrics.get('specificity', 0.0))
            test_aucs.append(test_metrics.get('auc', 0.0))
            fold_resource_records.append(fold_resources)

        summarize_results(test_accs, test_bals, test_sens, test_spes, test_aucs)
        total_resources = total_tracker.stop()
        for cuda_key in ['cuda_max_allocated_mb', 'cuda_max_reserved_mb']:
            fold_values = [item.get(cuda_key) for item in fold_resource_records if item.get(cuda_key) is not None]
            if fold_values:
                total_resources[cuda_key] = max(fold_values)
        report_path = save_resource_report(run_dir, {
            'run_dir': run_dir,
            'dataset': args.dataset,
            'construction_method': args.construction_method,
            'simplex_order': args.simplex_order,
            'num_folds': args.num_folds,
            'fold_limit': fold_limit,
            'folds': fold_resource_records,
            'total': total_resources,
        })
        print(f"Total Resource: {format_resource_summary(total_resources)}")
        print(f"Resource report saved to {report_path}")
    finally:
        sys.stdout = original_stdout


if __name__ == '__main__':
    main()
