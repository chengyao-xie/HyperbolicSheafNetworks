import json
import os
import time
from collections import Counter
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from models.cont_models import HyperbolicGraphDiffusion
from utils.resource_monitor import ResourceTracker, format_resource_summary


def custom_collate(data_list):
    if len(data_list) == 0:
        return None

    sigma1_list = []
    hodge_laplacian_list = []
    num_triangles_list = []
    num_edges_list = []
    cleaned_list = []

    for data in data_list:
        sigma1_list.append(data.sigma1 if hasattr(data, 'sigma1') else None)
        hodge_laplacian_list.append(
            data.hodge_laplacian if hasattr(data, 'hodge_laplacian') else None)
        num_triangles_list.append(data.num_triangles if hasattr(data, 'num_triangles') else None)
        num_edges_list.append(data.num_edges if hasattr(data, 'num_edges') else None)

        # Construct the lightweight graph view directly. Cloning the original
        # Data object also cloned the large boundary/Hodge tensors before they
        # were immediately removed.
        data_copy = data.__class__()
        excluded = {'sigma1', 'hodge_laplacian', 'num_triangles', 'num_edges'}
        for key, value in data:
            if key not in excluded:
                data_copy[key] = value
        cleaned_list.append(data_copy)

    batch = Batch.from_data_list(cleaned_list)
    batch.sigma1_list = sigma1_list
    batch.hodge_laplacian_list = hodge_laplacian_list
    batch.num_triangles_list = num_triangles_list
    batch.num_edges_list = num_edges_list
    return batch


def simple_collate(data_list):
    if len(data_list) == 0:
        return None
    return Batch.from_data_list(data_list)


def set_seed(seed, deterministic=True, benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def resolve_amp(args):
    if args.device.type != 'cuda' or args.amp == 'off':
        return False, None, None

    if args.amp == 'bf16':
        dtype = torch.bfloat16
    elif args.amp == 'fp16':
        dtype = torch.float16
    elif hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    return True, dtype, scaler


def autocast_context(device, enabled, dtype):
    if not enabled:
        return nullcontext()
    if hasattr(torch, 'autocast'):
        return torch.autocast(device_type=device.type, dtype=dtype)
    if device.type == 'cuda':
        return torch.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def move_to_device(data, device, non_blocking=False):
    return data.to(device, non_blocking=non_blocking)


def train_epoch(model, loader, optimizer, device, class_weights=None, grad_clip=None,
                amp_enabled=False, amp_dtype=None, scaler=None, accumulation_steps=1,
                non_blocking=False):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    accumulation_steps = max(1, accumulation_steps)
    optimizer.zero_grad(set_to_none=True)

    for step, data in enumerate(loader):
        data = move_to_device(data, device, non_blocking=non_blocking)

        with autocast_context(device, amp_enabled, amp_dtype):
            out = model(data)
            loss = F.nll_loss(out, data.y, weight=class_weights)

        scaled_loss = loss / accumulation_steps
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader)
        if should_step:
            if grad_clip:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        pred = out.argmax(dim=1)
        total_loss += loss.item()
        total_correct += (pred == data.y).sum().item()
        total_samples += data.y.size(0)

    return total_loss / len(loader), total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled=False, amp_dtype=None, non_blocking=False):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for data in loader:
        data = move_to_device(data, device, non_blocking=non_blocking)
        with autocast_context(device, amp_enabled, amp_dtype):
            out = model(data)
            loss = F.nll_loss(out, data.y)

        total_loss += loss.item()
        total_correct += (out.argmax(dim=1) == data.y).sum().item()
        total_samples += data.y.size(0)

        all_preds.append(out.argmax(dim=1).cpu())
        all_labels.append(data.y.cpu())
        all_probs.append(torch.exp(out).cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    sensitivity, specificity = classification_rates(all_labels, all_preds)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    auc = classification_auc(all_labels, all_probs)
    num_classes = all_probs.shape[1]
    pred_counts = np.bincount(all_preds, minlength=num_classes).astype(int).tolist()
    label_counts = np.bincount(all_labels, minlength=num_classes).astype(int).tolist()

    return {
        'loss': total_loss / len(loader),
        'accuracy': total_correct / total_samples,
        'balanced_accuracy': balanced_accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'pred_counts': pred_counts,
        'label_counts': label_counts,
    }, all_preds, all_labels


def classification_rates(labels, preds):
    cm = confusion_matrix(labels, preds)

    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sensitivity, specificity

    sens_list, spec_list = [], []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    return np.mean(sens_list), np.mean(spec_list)


def classification_auc(labels, probs):
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0
    if len(unique_labels) == 2:
        return roc_auc_score(labels, probs[:, 1])
    return roc_auc_score(labels, probs, multi_class='ovr', average='macro')


def build_loader(dataset, batch_size, shuffle, collate_fn, args):
    pin_memory = args.pin_memory and args.device.type == 'cuda'
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
        'num_workers': args.num_workers,
        'pin_memory': pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = args.persistent_workers
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def build_scheduler(optimizer, args, train_loader):
    if args.scheduler == 'none':
        return None
    if args.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=max(1, args.early_stopping // 3))


def step_scheduler(scheduler, args, train_loss):
    if scheduler is None:
        return
    if args.scheduler == 'plateau':
        scheduler.step(train_loss)
    else:
        scheduler.step()


def metric_value(metrics, metric_name):
    return metrics[metric_name]


def is_better(score, best_score, metric_name, min_delta):
    if best_score is None:
        return True
    if metric_name == 'loss':
        return score < best_score - min_delta
    return score > best_score + min_delta


def checkpoint_state(model, epoch, test_metrics):
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    return {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'test_metrics': test_metrics,
    }


def run_fold(args, dataset, fold_idx, train_idx, test_idx, run_dir):
    print(f"Fold {fold_idx + 1}/{args.num_folds}")
    fold_tracker = ResourceTracker(f'fold_{fold_idx + 1}', args.device).start()

    train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]
    train_labels = [data.y.item() for data in train_dataset]
    train_dist = Counter(train_labels)

    print(f"Split: Train={len(train_dataset)} | Test={len(test_dataset)}")

    class_counts = [train_dist[i] for i in range(dataset.num_classes)]
    class_weights = torch.tensor(
        [len(train_labels) / (dataset.num_classes * c) for c in class_counts],
        dtype=torch.float32, device=args.device,
    )

    collate_fn = custom_collate if args.simplex_order == 2 else simple_collate
    train_loader = build_loader(train_dataset, args.batch_size, True, collate_fn, args)
    test_loader = build_loader(test_dataset, args.batch_size, False, collate_fn, args)

    sample_data = dataset[0]
    graph_size = sample_data.x.size(0)
    model_args = vars(args).copy()
    model_args.update({
        'input_dim': sample_data.x.size(1),
        'output_dim': dataset.num_classes,
        'graph_size': graph_size,
    })

    model = HyperbolicGraphDiffusion(sample_data.edge_index, model_args).to(args.device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, train_loader)
    compile_enabled = args.compile_model and hasattr(torch, 'compile')
    if compile_enabled:
        model = torch.compile(model, mode=args.compile_mode)
    elif args.compile_model:
        print("Speed: torch.compile requested but unavailable in this PyTorch version")

    amp_enabled, amp_dtype, scaler = resolve_amp(args)
    non_blocking = args.pin_memory and args.device.type == 'cuda'
    amp_label = amp_dtype if amp_dtype is None else str(amp_dtype).replace('torch.', '')
    print(f"Loader: workers={args.num_workers} | pin_memory={args.pin_memory} | "
          f"accumulation={args.accumulation_steps}")
    print(f"Speed: amp={args.amp}({amp_label}) | compile={compile_enabled} | "
          f"scheduler={args.scheduler} | eval_interval={args.eval_interval}")
    print(f"Selection: best_test_{args.checkpoint_metric} | save_model={args.save_model}")

    best_test_metrics = {
        'loss': float('inf'),
        'accuracy': 0,
        'balanced_accuracy': 0,
        'sensitivity': 0,
        'specificity': 0,
        'auc': 0,
        'pred_counts': [],
        'label_counts': [],
    }
    best_score = None
    best_epoch = -1

    epoch_times = []
    epoch_history = []
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        learning_rate = float(optimizer.param_groups[0]['lr'])
        train_start = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device, class_weights, args.grad_clip,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
            accumulation_steps=args.accumulation_steps, non_blocking=non_blocking)
        train_seconds = time.perf_counter() - train_start
        step_scheduler(scheduler, args, train_loss)

        should_eval = (
            epoch == 0
            or (epoch + 1) % max(1, args.eval_interval) == 0
            or (epoch + 1) == args.epochs
        )
        test_metrics = None
        eval_seconds = 0.0
        improved = False
        if should_eval:
            eval_start = time.perf_counter()
            test_metrics, _, _ = evaluate(
                model, test_loader, args.device,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype, non_blocking=non_blocking)
            eval_seconds = time.perf_counter() - eval_start
            current_score = metric_value(test_metrics, args.checkpoint_metric)
            improved = is_better(current_score, best_score, args.checkpoint_metric, args.min_delta)
            if improved:
                best_score = current_score
                best_epoch = epoch
                best_test_metrics = test_metrics.copy()
                if args.save_model:
                    torch.save(
                        checkpoint_state(model, epoch, test_metrics),
                        os.path.join(run_dir, f'best_model_fold{fold_idx}.pt'),
                    )

        epoch_seconds = time.perf_counter() - epoch_start
        epoch_times.append(epoch_seconds)
        epoch_history.append({
            'epoch': epoch,
            'epoch_number': epoch + 1,
            'learning_rate': learning_rate,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'train_seconds': float(train_seconds),
            'eval_seconds': float(eval_seconds),
            'epoch_seconds': float(epoch_seconds),
            'is_best': bool(improved),
            'test': test_metrics.copy() if test_metrics is not None else None,
        })

        should_log = (
            epoch == 0
            or improved
            or (epoch + 1) % max(1, args.log_interval) == 0
            or (epoch + 1) == args.epochs
        )
        if should_log:
            if test_metrics is None:
                test_text = "Test: skipped"
            else:
                marker = "*" if improved else ""
                test_text = (
                    f"Test{marker}: acc={test_metrics['accuracy']:.4f} "
                    f"bal={test_metrics['balanced_accuracy']:.4f} "
                    f"auc={test_metrics['auc']:.4f} "
                    f"sen/spe={test_metrics['sensitivity']:.4f}/{test_metrics['specificity']:.4f} "
                    f"pred={test_metrics['pred_counts']} label={test_metrics['label_counts']}"
                )
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"{test_text} | Time: {epoch_seconds:.2f}s "
                  f"(train {train_seconds:.2f}s, eval {eval_seconds:.2f}s)")

        if best_epoch >= 0 and args.early_stopping > 0 and epoch - best_epoch >= args.early_stopping:
            print(f"Early stopping at epoch {epoch}: no {args.checkpoint_metric} improvement "
                  f"for {args.early_stopping} epochs")
            break

    history_path = os.path.join(run_dir, f'metrics_fold{fold_idx}.json')
    with open(history_path, 'w', encoding='utf-8') as history_file:
        json.dump({
            'fold': fold_idx,
            'selection_metric': args.checkpoint_metric,
            'best_epoch': best_epoch,
            'best_epoch_number': best_epoch + 1,
            'best_test_metrics': best_test_metrics,
            'model_saved': bool(args.save_model),
            'epochs': epoch_history,
        }, history_file, indent=2, ensure_ascii=False)

    print(f"\nBest Test (epoch {best_epoch}, number {best_epoch + 1}): "
          f"Acc={best_test_metrics['accuracy']:.4f} | "
          f"Bal={best_test_metrics['balanced_accuracy']:.4f} | "
          f"Sen={best_test_metrics['sensitivity']:.4f} | Spe={best_test_metrics['specificity']:.4f} | "
          f"AUC={best_test_metrics['auc']:.4f} | Pred={best_test_metrics['pred_counts']} | "
          f"Label={best_test_metrics['label_counts']}")
    print(f"Epoch metrics saved to {history_path}")

    fold_resources = fold_tracker.stop()
    fold_resources['epoch_time_mean_seconds'] = float(np.mean(epoch_times)) if epoch_times else 0.0
    fold_resources['epoch_time_std_seconds'] = float(np.std(epoch_times)) if epoch_times else 0.0
    fold_resources['epochs'] = len(epoch_times)
    fold_resources['best_epoch'] = best_epoch
    print(f"Fold Resource: {format_resource_summary(fold_resources)}")

    return best_test_metrics['accuracy'], best_test_metrics, fold_resources


def build_optimizer(model, args):
    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer_cls = torch.optim.AdamW if args.optimizer == 'adamw' else torch.optim.Adam
    return optimizer_cls([
        {'params': sheaf_learner_params, 'weight_decay': args.sheaf_decay},
        {'params': other_params, 'weight_decay': args.weight_decay},
    ], lr=args.lr)
