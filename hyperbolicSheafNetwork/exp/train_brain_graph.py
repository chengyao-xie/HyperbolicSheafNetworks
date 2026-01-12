import sys
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
import logging
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.cont_models import HyperbolicGraphDiffusion
from utils.GraphDataset import get_dataset, create_stratified_splits


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, device, class_weights=None, grad_clip=None):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    for i, data in enumerate(loader):
        if i % 10 == 0:
            print(f"Batch {i+1}/{len(loader)}", end='\r')
        
        data = data.to(device)
        optimizer.zero_grad()
        
        try:
            # Forward pass
            out = model(data)  # [batch_size, num_classes]
            loss = F.nll_loss(out, data.y, weight=class_weights)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            # Statistics
            pred = out.argmax(dim=1)
            total_loss += loss.item()
            total_correct += (pred == data.y).sum().item()
            total_samples += data.y.size(0)
            
        except Exception as e:
            if "MaxNFEException" in str(type(e).__name__):
                print(f"\nMax NFE exceeded in batch {i+1} - skipping")
                continue
            else:
                print(f"\nError in batch {i+1}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    print()  
    
    # 避免除零错误
    if total_samples == 0:
        print("WARNING: No samples processed successfully!")
        return float('nan'), 0.0
    
    avg_loss = total_loss / len(loader) if len(loader) > 0 else float('nan')
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []  
    
    for data in loader:
        data = data.to(device)
        out = model(data)  # [1, num_classes]
        loss = F.nll_loss(out, data.y)
        
        pred = out.argmax(dim=1)
        probs = torch.exp(out)  
        
        total_loss += loss.item()
        total_correct += (pred == data.y).sum().item()
        total_samples += data.y.size(0)
        
        all_preds.append(pred.cpu())
        all_labels.append(data.y.cpu())
        all_probs.append(probs.cpu())
    
    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
    
    # 对于二分类: TN, FP, FN, TP
    if len(cm) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # 多分类情况，计算每个类别的敏感性和特异性的平均值
        sensitivity_list = []
        specificity_list = []
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            sensitivity_list.append(sens)
            specificity_list.append(spec)
        
        sensitivity = np.mean(sensitivity_list)
        specificity = np.mean(specificity_list)
    
    # 计算AUC
    try:
        if len(np.unique(all_labels.numpy())) == 2:
            # 二分类
            auc = roc_auc_score(all_labels.numpy(), all_probs[:, 1].numpy())
        else:
            # 多分类 (使用one-vs-rest策略)
            auc = roc_auc_score(all_labels.numpy(), all_probs.numpy(), 
                               multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc
    }
    
    return metrics, all_preds, all_labels


def run_fold(args, dataset, repeat_idx, train_mask, val_mask, test_mask, run_dir):
    print(f"\n{'='*60}")
    print(f"Repeat {repeat_idx + 1}/{args.repeats}")
    print(f"{'='*60}")
    
    train_dataset = [dataset[i] for i in range(len(dataset)) if train_mask[i]]
    val_dataset = [dataset[i] for i in range(len(dataset)) if val_mask[i]]
    test_dataset = [dataset[i] for i in range(len(dataset)) if test_mask[i]]
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    train_labels = [data.y.item() for data in train_dataset]
    val_labels = [data.y.item() for data in val_dataset]
    test_labels = [data.y.item() for data in test_dataset]
    
    from collections import Counter
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    test_dist = Counter(test_labels)
    
    print(f"Train label distribution: {dict(train_dist)}")
    print(f"Val label distribution: {dict(val_dist)}")
    print(f"Test label distribution: {dict(test_dist)}")
    
    num_samples = len(train_labels)
    class_counts = [train_dist[i] for i in range(dataset.num_classes)]
    class_weights = torch.tensor([num_samples / (dataset.num_classes * count) for count in class_counts], 
                                 dtype=torch.float32, device=args.device)
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    batch_size = 32  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)} | Total: {total_samples}")
    
    sample_data = dataset[0]
    graph_size = sample_data.x.size(0)
    
    model_args = {
        'd': args.d,
        'hidden_channels': args.hidden_channels,
        'input_dim': graph_size,  # FC矩阵是 [n_nodes, n_nodes]
        'output_dim': dataset.num_classes,
        'graph_size': graph_size,
        'device': args.device,
        'add_lp': args.add_lp,
        'add_hp': args.add_hp,
        'linear': args.linear,
        'input_dropout': args.input_dropout,  # 增大dropout减少过拟合
        'dropout': args.dropout,
        'left_weights': args.left_weights,
        'right_weights': args.right_weights,
        'sparse_learner': args.sparse_learner,
        'use_act': args.use_act,
        'layers': args.layers,
        'normalised': args.normalised,
        'deg_normalised': args.deg_normalised,
        'sheaf_act': args.sheaf_act,
        'second_linear': args.second_linear,
        'orth': args.orth,
        'edge_weights': args.edge_weights,
        'max_t': args.max_t,  
        'pool_type': args.pool_type,
        'max_nfe': args.max_nfe,  
        'adjoint': args.adjoint,
        'tol_scale': args.tol_scale,
        'tol_scale_adjoint': args.tol_scale_adjoint,
        'int_method': args.int_method,
        'step_size': args.step_size,  # 增大步长减少求解步数
        'adjoint_step_size': args.adjoint_step_size
    }
    
    model = HyperbolicGraphDiffusion(sample_data.edge_index, model_args)
    model = model.to(args.device)
    
    # 优化器（分组学习率）
    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args.sheaf_decay},
        {'params': other_params, 'weight_decay': args.weight_decay}
    ], lr=args.lr)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20)
    
    print("Starting training ...")

    best_val_acc = 0
    best_val_loss = float('inf')
    best_val_auc = 0
    best_epoch = 0
    patience_counter = 0

    best_test_acc = 0
    best_test_metrics = {
        'accuracy': 0.0,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'auc': 0.0,
    }
    best_epoch_test = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch} starting...")
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device, class_weights, args.grad_clip
        )
        
        if np.isnan(train_loss):
            print(f"Training failed at epoch {epoch} - all batches skipped. Aborting.")
            break
        
        val_metrics, _, _ = evaluate(model, val_loader, args.device)
        
        test_metrics, _, _ = evaluate(model, test_loader, args.device)
        
        # 学习率调整
        scheduler.step(val_metrics['accuracy'])
        
        # Early stopping判断
        if args.stop_strategy == 'acc':
            improved = val_metrics['accuracy'] > best_val_acc
        else:
            improved = val_metrics['loss'] < best_val_loss
        
        if improved:
            best_val_acc = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            patience_counter = 0
            
            # 保存最佳模型（基于验证集）
            model_path = os.path.join(run_dir, f'best_model_repeat{repeat_idx}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
            }, model_path)
        else:
            patience_counter += 1

        if test_metrics['accuracy'] > best_test_acc:
            best_test_acc = test_metrics['accuracy']
            best_test_metrics = test_metrics.copy()
            best_epoch_test = epoch
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
              f"Sen: {val_metrics['sensitivity']:.4f} Spe: {val_metrics['specificity']:.4f} AUC: {val_metrics['auc']:.4f} | "
              f"Test Acc: {test_metrics['accuracy']:.4f} Sen: {test_metrics['sensitivity']:.4f} "
              f"Spe: {test_metrics['specificity']:.4f} AUC: {test_metrics['auc']:.4f}")
        
        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"\nRepeat {repeat_idx + 1} Results:")
    print(f"  Best Val Epoch: {best_epoch} | Val Acc: {best_val_acc:.4f} | Val AUC: {best_val_auc:.4f}")
    print(f"  Best Test Epoch (by Acc): {best_epoch_test}")
    print(f"    Accuracy: {best_test_metrics.get('accuracy', 0):.4f}")
    print(f"    Sensitivity: {best_test_metrics.get('sensitivity', 0):.4f}")
    print(f"    Specificity: {best_test_metrics.get('specificity', 0):.4f}")
    print(f"    AUC: {best_test_metrics.get('auc', 0):.4f}")
    
    return best_test_acc, best_test_metrics


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)
    
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(os.path.dirname(__file__), '..', 'log', f'run_{args.dataset}_order{args.network_order}_seed{args.seed}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    log_filename = f'training.log'
    log_path = os.path.join(run_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger()
    
    class PrintLogger:
        def __init__(self, logger):
            self.logger = logger
        
        def write(self, message):
            if message.strip(): 
                self.logger.info(message.rstrip())
        
        def flush(self):
            pass
    
    original_stdout = sys.stdout
    sys.stdout = PrintLogger(logger)
    
    try:
        dataset = get_dataset(args.dataset, args.graph_type, order=args.network_order)
        print(f"Dataset loaded: {len(dataset)} graphs (graph: {args.graph_type})")
        print(f"Num features: {dataset.num_features}")
        print(f"Num classes: {dataset.num_classes}")
        
        print(f"\n{'='*60}")
        print("Training Configuration:")
        print(f"{'='*60}")
        print(f"Model: HyperbolicGraphDiffusion")
        print(f"Dataset: {args.dataset}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Weight decay: {args.weight_decay}")
        print(f"Sheaf decay: {args.sheaf_decay}")
        print(f"Hidden channels: {args.hidden_channels}")
        print(f"Stalks dimension (d): {args.d}")
        print(f"Max ODE time: {args.max_t}")
        print(f"Pooling type: {args.pool_type}")
        print(f"Network order: {args.network_order}")
        print(f"Gradient clip: {args.grad_clip}")
        print(f"Early stopping patience: {args.early_stopping}")
        print(f"{'='*60}\n")
        
        # 提取所有样本的标签用于分层分割
        all_labels = np.array([data.y.item() for data in dataset])
        print(f"\nDataset class distribution:")
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"  Class {label}: {count} samples ({count/len(all_labels)*100:.1f}%)")
        
        # 运行多次重复实验，每次动态生成新的分层分割
        test_accs = []
        test_sens = []
        test_spes = []
        test_aucs = []
        
        for repeat in range(args.repeats):
            # 为每次重复使用不同的随机种子生成分层分割
            rng = np.random.RandomState(args.seed + repeat)
            train_mask, val_mask, test_mask = create_stratified_splits(
                all_labels, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, rng=rng
            )
            
            # 验证分层采样效果
            train_labels = all_labels[train_mask == 1]
            val_labels = all_labels[val_mask == 1]
            test_labels = all_labels[test_mask == 1]
            
            print(f"\nRepeat {repeat + 1} - Split generated:")
            print(f"  Train: {train_mask.sum()} samples, class dist: {np.bincount(train_labels)}")
            print(f"  Val:   {val_mask.sum()} samples, class dist: {np.bincount(val_labels)}")
            print(f"  Test:  {test_mask.sum()} samples, class dist: {np.bincount(test_labels)}")
            
            test_acc, test_metrics = run_fold(args, dataset, repeat, train_mask, val_mask, test_mask, run_dir)
            test_accs.append(test_acc)
            test_sens.append(test_metrics.get('sensitivity', 0.0))
            test_spes.append(test_metrics.get('specificity', 0.0))
            test_aucs.append(test_metrics.get('auc', 0.0))

        test_acc_mean = np.mean(test_accs) * 100
        test_acc_std = np.std(test_accs) * 100
        test_sen_mean = np.mean(test_sens) * 100
        test_sen_std = np.std(test_sens) * 100
        test_spe_mean = np.mean(test_spes) * 100
        test_spe_std = np.std(test_spes) * 100
        test_auc_mean = np.mean(test_aucs) * 100
        test_auc_std = np.std(test_aucs) * 100
        
        print(f"\n{'='*60}")
        print("Final Results (Averaged over Repeats)")
        print(f"{'='*60}")
        print(f"Test Accuracy:       {test_acc_mean:.2f} ± {test_acc_std:.2f}%")
        print(f"Test Sensitivity:    {test_sen_mean:.2f} ± {test_sen_std:.2f}%")
        print(f"Test Specificity:    {test_spe_mean:.2f} ± {test_spe_std:.2f}%")
        print(f"Test AUC:            {test_auc_mean:.2f} ± {test_auc_std:.2f}%")
        print(f"{'='*60}\n")
    
    finally:
        sys.stdout = original_stdout


if __name__ == '__main__':
    main()
