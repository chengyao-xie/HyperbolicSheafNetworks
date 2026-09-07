import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from models.cont_models import HyperbolicGraphDiffusion
def custom_collate(data_list):
    fields = (
        'sigma1', 'sigma2', 'edge_node_index', 'edge_node_sign',
        'face_edge_index', 'face_edge_sign', 'num_triangles', 'num_edges',
    )
    values = {field: [] for field in fields}
    graphs = []
    for graph in data_list:
        graph_copy = graph.clone()
        for field in fields:
            value = getattr(graph, field)
            values[field].append(value.clone() if torch.is_tensor(value) else value)
            delattr(graph_copy, field)
        graphs.append(graph_copy)

    batch = Batch.from_data_list(graphs)
    for field, field_values in values.items():
        setattr(batch, f'{field}_list', field_values)
    return batch


def simple_collate(data_list):
    return Batch.from_data_list(data_list)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    return torch.autocast(device_type=device.type, dtype=dtype)


def train_epoch(model, loader, optimizer, device, grad_clip=None,
                amp_enabled=False, amp_dtype=None, scaler=None,
                non_blocking=False):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for data in loader:
        data = data.to(device, non_blocking=non_blocking)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled, amp_dtype):
            out = model(data)
            loss = F.nll_loss(out, data.y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if grad_clip:
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if scaler is not None and scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

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
        data = data.to(device, non_blocking=non_blocking)
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
    """
    For binary tasks this matches the mean of sensitivity/specificity.  For
    PPMI's four classes, use the standard macro recall definition instead of
    averaging one-vs-rest specificity into the reported balanced accuracy.
    """
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
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
    }, all_preds, all_labels, all_probs


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
    pin_memory = args.device.type == 'cuda'
    if pin_memory and dataset:
        sample = dataset[0]
        for name in ('sigma1', 'sigma2'):
            boundary = getattr(sample, name, None)
            if boundary is not None and boundary.layout != torch.strided:
                pin_memory = False
                break
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'collate_fn': collate_fn,
        'num_workers': args.num_workers,
        'pin_memory': pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    return DataLoader(dataset, **loader_kwargs)


def run_fold(args, dataset, fold_idx, train_idx, test_idx, run_dir):
    print(f"Fold {fold_idx + 1}/{args.num_folds}")

    outer_train_dataset = [dataset[i] for i in train_idx]
    test_dataset = [dataset[i] for i in test_idx]
    validation_dataset = None
    if args.validation_fraction > 0:
        outer_labels = np.asarray([data.y.item() for data in outer_train_dataset])
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=args.validation_fraction,
            random_state=args.seed + fold_idx)
        fit_positions, val_positions = next(
            splitter.split(np.zeros(len(outer_labels)), outer_labels))
        train_dataset = [outer_train_dataset[i] for i in fit_positions]
        validation_dataset = [outer_train_dataset[i] for i in val_positions]
    else:
        train_dataset = outer_train_dataset
    split_text = f"Split: Train={len(train_dataset)}"
    if validation_dataset is not None:
        split_text += f" | Val={len(validation_dataset)}"
    print(split_text + f" | Test={len(test_dataset)}")

    collate_fn = custom_collate if args.simplex_order == 2 else simple_collate
    train_loader = build_loader(train_dataset, args.batch_size, True, collate_fn, args)
    validation_loader = (
        build_loader(validation_dataset, args.batch_size, False, collate_fn, args)
        if validation_dataset is not None else None)
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

    amp_enabled, amp_dtype, scaler = resolve_amp(args)
    non_blocking = args.device.type == 'cuda'
    amp_label = amp_dtype if amp_dtype is None else str(amp_dtype).replace('torch.', '')
    print(f"Loader: workers={args.num_workers} | pin_memory={non_blocking}")
    print(f"Speed: amp={args.amp}({amp_label})")

    best_selection_metrics = {
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

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device, args.grad_clip,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler,
            non_blocking=non_blocking)
        selection_loader = validation_loader if validation_loader is not None else test_loader
        selection_metrics, _, _, _ = evaluate(
            model, selection_loader, args.device,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, non_blocking=non_blocking)
        current_score = selection_metrics['loss']
        improved = best_score is None or current_score < best_score
        if improved:
            best_score = current_score
            best_epoch = epoch
            best_selection_metrics = selection_metrics.copy()
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                },
                os.path.join(run_dir, f'best_model_fold{fold_idx}.pt'),
            )

        marker = "*" if improved else ""
        selection_name = "Val" if validation_dataset is not None else "Test"
        selection_text = (
            f"{selection_name}{marker}: acc={selection_metrics['accuracy']:.4f} "
            f"bal={selection_metrics['balanced_accuracy']:.4f} "
            f"auc={selection_metrics['auc']:.4f} "
            f"sen/spe={selection_metrics['sensitivity']:.4f}/{selection_metrics['specificity']:.4f} "
            f"pred={selection_metrics['pred_counts']} label={selection_metrics['label_counts']}"
        )
        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"{selection_text}")

        if best_epoch >= 0 and args.early_stopping > 0 and epoch - best_epoch >= args.early_stopping:
            print(f"Early stopping at epoch {epoch}: no loss improvement "
                  f"for {args.early_stopping} epochs")
            break

    if validation_dataset is not None and best_epoch >= 0:
        checkpoint_path = os.path.join(run_dir, f'best_model_fold{fold_idx}.pt')
        state = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model_to_restore = getattr(model, '_orig_mod', model)
        model_to_restore.load_state_dict(state['model_state_dict'])
        best_test_metrics, _, _, _ = evaluate(
            model, test_loader, args.device,
            amp_enabled=amp_enabled, amp_dtype=amp_dtype, non_blocking=non_blocking)
        print(f"\nBest Val (epoch {best_epoch}): Acc={best_selection_metrics['accuracy']:.4f} | "
              f"Bal={best_selection_metrics['balanced_accuracy']:.4f} | "
              f"AUC={best_selection_metrics['auc']:.4f}")
        print(f"Test at Best Val: Acc={best_test_metrics['accuracy']:.4f} | "
              f"Bal={best_test_metrics['balanced_accuracy']:.4f} | "
              f"Sen={best_test_metrics['sensitivity']:.4f} | Spe={best_test_metrics['specificity']:.4f} | "
              f"AUC={best_test_metrics['auc']:.4f} | Pred={best_test_metrics['pred_counts']} | "
              f"Label={best_test_metrics['label_counts']}")
    else:
        best_test_metrics = best_selection_metrics
        print(f"\nBest Test: Acc={best_test_metrics['accuracy']:.4f} | "
              f"Bal={best_test_metrics['balanced_accuracy']:.4f} | "
              f"Sen={best_test_metrics['sensitivity']:.4f} | Spe={best_test_metrics['specificity']:.4f} | "
              f"AUC={best_test_metrics['auc']:.4f} | Pred={best_test_metrics['pred_counts']} | "
              f"Label={best_test_metrics['label_counts']}")

    selection_metadata = {
        'selection_split': 'validation' if validation_dataset is not None else 'test',
        'fit_size': len(train_dataset),
        'validation_size': len(validation_dataset) if validation_dataset is not None else 0,
        'test_size': len(test_dataset),
        'best_epoch': best_epoch,
        'checkpoint_metric': 'loss',
        'test_fold_used_for_selection': validation_dataset is None,
    }
    return best_test_metrics['accuracy'], best_test_metrics, selection_metadata


def build_optimizer(model, args):
    sheaf_learner_params, other_params = model.grouped_parameters()
    return torch.optim.AdamW([
        {'params': sheaf_learner_params, 'weight_decay': args.sheaf_decay},
        {'params': other_params, 'weight_decay': args.weight_decay},
    ], lr=args.lr)
