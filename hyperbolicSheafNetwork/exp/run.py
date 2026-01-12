#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
import numpy as np
import wandb
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.cont_models import HyperbolicGraphClassifier
from utils.heterophilic import get_dataset


# ========== 数据处理函数 ==========

def extract_graph_labels(data):
    if hasattr(data, 'graph_label'):
        return data.graph_label
    elif hasattr(data, 'graph_labels'):
        return data.graph_labels
    elif hasattr(data, 'y') and data.y.numel() == 1:
        return data.y
    elif hasattr(data, 'y'):
        return data.y[0] if data.y.numel() > 1 else data.y
    else:
        raise ValueError("No labels found in data")


def prepare_graph_data(graph_data_item, device, add_noise=False, noise_scale=0.01):
    """准备单个图数据，可选添加噪声增强"""
    graph_data_item = graph_data_item.to(device)
    if add_noise:
        noise = torch.randn_like(graph_data_item.x) * noise_scale
        graph_data_item.x = graph_data_item.x + noise
    return graph_data_item


def normalize_labels(labels, device):
    """标准化标签格式"""
    if isinstance(labels, (int, float)):
        return torch.tensor(labels, dtype=torch.float32, device=device)
    elif labels.dim() > 0 and labels.numel() == 1:
        return labels.float().squeeze()
    else:
        return labels.float()


def normalize_logits(logits):
    """标准化模型输出格式"""
    if logits.dim() > 0:
        if logits.numel() == 1:
            return logits.squeeze()
        elif logits.size(-1) == 1:
            logits = logits.squeeze(-1)
            if logits.dim() > 0 and logits.numel() == 1:
                return logits.squeeze()
    return logits


# ========== 训练和评估函数 ==========

class FocalLoss(torch.nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_graph_classification_loss(model, data):
    """计算图分类损失"""
    labels = extract_graph_labels(data)
    labels = normalize_labels(labels, data.x.device)
    
    out = model(data.x)
    out = normalize_logits(out)
    
    num_classes = getattr(model, 'num_classes', 2)
    
    if num_classes == 2:
        # 使用模型内置的criterion（Focal Loss）
        return model.criterion(out, labels)
    else:
        if out.dim() == 0:
            out = out.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        return F.cross_entropy(out, labels.long())


def batch_training(model, batch_graphs, device, epoch, optimizer, grad_clip=1.0, scaler=None):
    """批量训练，逐图前向/反传/更新"""
    batch_loss = 0.0
    for graph_data_item in batch_graphs:
        graph_data_item = prepare_graph_data(graph_data_item, device, add_noise=(epoch > 10), noise_scale=0.01)
        
        use_amp = (scaler is not None) and scaler.is_enabled()
        # 使用新版API，兼容PyTorch 1.10+
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss = compute_graph_classification_loss(model, graph_data_item)
        
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        batch_loss += loss.detach()
    
    return batch_loss / max(1, len(batch_graphs))


def test_graph_classification(model, data):
    """测试单个图的分类结果"""
    model.eval()
    with torch.no_grad():
        labels = extract_graph_labels(data)
        labels = normalize_labels(labels, data.x.device)
        
        logits = model(data.x)
        logits = normalize_logits(logits)
        
        num_classes = getattr(model, 'num_classes', 2)
        
        if num_classes == 2:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()
        else:
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            loss = F.cross_entropy(logits, labels.long())
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
        
        # 确保所有返回值都至少是1维的
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
            
        acc = (pred == labels).float().mean().item()
        return acc, loss.item(), pred.detach().cpu(), probs.detach().cpu(), labels.detach().cpu()


def evaluate_graph_classification(model, data_list, device):
    """评估图分类模型，返回acc, loss, auc, sensitivity, specificity"""
    model.eval()
    total_correct = 0
    total_loss = 0
    total_samples = len(data_list)
    
    if total_samples == 0:
        return 0.0, float('inf'), 0.0, 0.0, 0.0
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for graph_data in data_list:
            graph_data = graph_data.to(device)
            acc, loss, pred, probs, labels = test_graph_classification(model, graph_data)
            total_correct += acc
            total_loss += loss
            all_preds.append(pred)
            all_probs.append(probs)
            all_labels.append(labels)
    
    # 转换为numpy数组进行指标计算
    all_preds = torch.cat(all_preds).numpy().flatten()
    all_probs = torch.cat(all_probs).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()
    
    # 计算AUC
    try:
        if len(np.unique(all_labels)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        auc = 0.0
    
    # 计算Sensitivity和Specificity（针对二分类）
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    tn = np.sum((all_preds == 0) & (all_labels == 0))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return total_correct / total_samples, total_loss / total_samples, auc, sensitivity, specificity


# ========== 主实验函数 ==========


def run_exp(args, dataset, model_cls, fold):
    if len(dataset) < 10:
        print(f"Warning: Dataset only has {len(dataset)} graphs, which is too small for proper train/val/test split")
        print("Using simple split for demonstration...")
    
    total_graphs = len(dataset)
    train_size = max(1, int(total_graphs * 0.6))
    val_size = max(1, int(total_graphs * 0.2))
    test_size = total_graphs - train_size - val_size
    
    # 根据fold创建不同的随机划分
    torch.manual_seed(42 + fold)
    indices = torch.randperm(total_graphs)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    print(f"Graph split - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    test_data = [dataset[i] for i in test_indices]
    
    # 打印数据集标签分布
    train_labels = [extract_graph_labels(d).item() for d in train_data]
    val_labels = [extract_graph_labels(d).item() for d in val_data]
    test_labels = [extract_graph_labels(d).item() for d in test_data]
    
    print(f"Train label distribution: {np.bincount(train_labels)}")
    print(f"Val label distribution: {np.bincount(val_labels)}")
    print(f"Test label distribution: {np.bincount(test_labels)}")
    
    # 计算类别权重和初始偏置
    train_label_counts = np.bincount(train_labels)
    if len(train_label_counts) == 2:
        n_neg = train_label_counts[0]
        n_pos = train_label_counts[1]
        pos_weight = n_neg / (n_pos + 1e-8)
        
        # 计算初始偏置：ln(n_pos / n_neg)
        init_bias = np.log((n_pos + 1) / (n_neg + 1))
        args['pos_weight'] = pos_weight
        args['init_bias'] = init_bias
        
        # 使用Focal Loss参数
        args['focal_alpha'] = 0.75  # 更关注正类
        args['focal_gamma'] = 2.0
        
        print(f"[INFO] Class imbalance: n_neg={n_neg}, n_pos={n_pos}")
        print(f"[INFO] pos_weight={pos_weight:.4f}, init_bias={init_bias:.4f}")
        print(f"[INFO] Using Focal Loss with alpha={args['focal_alpha']}, gamma={args['focal_gamma']}")
    
    sample_data = dataset[0].to(args['device'])

    model = model_cls(sample_data.edge_index, args)
    model = model.to(args['device'])
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")
    
    # AMP scaler
    use_amp = bool(args.get('use_amp', False) and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp)

    sheaf_learner_params, other_params = model.grouped_parameters()
    
    base_lr = 0.005  # 提高学习率
    optimizer = torch.optim.AdamW([
        {'params': sheaf_learner_params, 'weight_decay': 5e-4, 'lr': base_lr * 2.0},  # 降低正则化
        {'params': other_params, 'weight_decay': 1e-3, 'lr': base_lr}
    ], betas=(0.9, 0.999), eps=1e-8)
    
    # 使用StepLR替代CosineAnnealing，更简单直接
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=30,
        gamma=0.5
    )
    
    epoch = 0
    best_val_acc = test_acc = best_tmp_test_acc = 0
    best_val_loss = float('inf')
    best_tmp_test_loss = float('inf')
    best_val_auc = best_tmp_test_auc = 0.0
    best_val_sen = best_tmp_test_sen = 0.0
    best_val_spe = best_tmp_test_spe = 0.0
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0
    
    # 添加学习率衰减和更灵活的早停策略
    patience = args.get('early_stopping', 50)  # 增加patience
    min_delta = 0.001  # 最小改进阈值

    for epoch in range(args['epochs']):
        # 移除学习率预热，直接使用正常学习率
        
        # 图分类训练 - 使用类别平衡采样
        model.train()
        running_loss = 0.0
        
        # 类别平衡采样：确保每个batch包含两个类别
        pos_samples = [d for d in train_data if extract_graph_labels(d).item() == 1]
        neg_samples = [d for d in train_data if extract_graph_labels(d).item() == 0]
        
        max_train_graphs = min(50, len(train_data))
        n_pos_batch = min(len(pos_samples), max_train_graphs // 2)
        n_neg_batch = max_train_graphs - n_pos_batch
        
        pos_subset = random.sample(pos_samples, n_pos_batch) if len(pos_samples) > 0 else []
        neg_subset = random.sample(neg_samples, n_neg_batch) if len(neg_samples) > 0 else []
        train_subset = pos_subset + neg_subset
        random.shuffle(train_subset)
        
        batch_size = min(4, len(train_subset))  # 减小batch size
        num_batches = 0
        
        for i in range(0, len(train_subset), batch_size):
            batch_graphs = train_subset[i:i + batch_size]
            batch_loss = batch_training(model, batch_graphs, args['device'], epoch, optimizer, args['grad_clip'], scaler)
            running_loss += float(batch_loss.item() if torch.is_tensor(batch_loss) else batch_loss)
            num_batches += 1
        avg_loss = running_loss / max(1, num_batches)
        
        # 每个epoch都评估
        train_sample = train_subset[:min(16, len(train_subset))]
        val_sample = val_data[:min(16, len(val_data))]
        test_sample = test_data[:min(16, len(test_data))]
        
        train_acc, train_loss, train_auc, train_sen, train_spe = evaluate_graph_classification(model, train_sample, args['device'])
        val_acc, val_loss, val_auc, val_sen, val_spe = evaluate_graph_classification(model, val_sample, args['device'])
        tmp_test_acc, tmp_test_loss, tmp_test_auc, tmp_test_sen, tmp_test_spe = evaluate_graph_classification(model, test_sample, args['device'])
                
        # 添加梯度和权重统计
        if epoch % 5 == 0:
            grad_norm = 0.0
            weight_norm = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
                weight_norm += p.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            weight_norm = weight_norm ** 0.5
            print(f"[Epoch {epoch}] Grad norm: {grad_norm:.6f}, Weight norm: {weight_norm:.6f}")
            
            # 检查模型输出
            model.eval()
            with torch.no_grad():
                sample_out = model(train_sample[0].x.to(args['device']))
                print(f"[Epoch {epoch}] Sample output: {sample_out.item():.6f}")
            model.train()
        
        scheduler.step()  # 在每个epoch后更新学习率
        
        # 每5个epoch打印详细信息
        if epoch % 5 == 0:
            # 手动打印学习率变化
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[INFO] Current learning rate: {current_lr:.6f}")
            
            # 检查模型输出分布
            model.eval()
            with torch.no_grad():
                sample_outputs = []
                for i in range(min(4, len(train_subset))):
                    out = model(train_subset[i].x.to(args['device']))
                    sample_outputs.append(out.item())
                print(f"[Epoch {epoch}] Sample outputs: {sample_outputs}")
                print(f"[Epoch {epoch}] Output mean: {np.mean(sample_outputs):.4f}, std: {np.std(sample_outputs):.4f}")
            model.train()
        
        print(f"[Epoch {epoch}]")
        print(f"Train: acc={train_acc:.4f}, loss={train_loss:.4f}, auc={train_auc:.4f}, sen={train_sen:.4f}, spe={train_spe:.4f}")
        print(f"Val:   acc={val_acc:.4f}, loss={val_loss:.4f}, auc={val_auc:.4f}, sen={val_sen:.4f}, spe={val_spe:.4f}")
        print(f"Test:  acc={tmp_test_acc:.4f}, loss={tmp_test_loss:.4f}, auc={tmp_test_auc:.4f}, sen={tmp_test_sen:.4f}, spe={tmp_test_spe:.4f}")

        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_train_auc': train_auc,
                f'fold{fold}_train_sen': train_sen,
                f'fold{fold}_train_spe': train_spe,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_val_auc': val_auc,
                f'fold{fold}_val_sen': val_sen,
                f'fold{fold}_val_spe': val_spe,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
                f'fold{fold}_tmp_test_auc': tmp_test_auc,
                f'fold{fold}_tmp_test_sen': tmp_test_sen,
                f'fold{fold}_tmp_test_spe': tmp_test_spe,
            }
            wandb.log(res_dict, step=epoch)

        new_best_trigger = val_acc > best_val_acc + min_delta if args['stop_strategy'] == 'acc' else val_loss < best_val_loss - min_delta
        if new_best_trigger:
            best_tmp_test_acc = tmp_test_acc
            best_tmp_test_loss = tmp_test_loss
            best_tmp_test_auc = tmp_test_auc
            best_tmp_test_sen = tmp_test_sen
            best_tmp_test_spe = tmp_test_spe
            test_acc = tmp_test_acc
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_val_sen = val_sen
            best_val_spe = val_spe
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1


    print(f"\nFold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}, AUC: {best_tmp_test_auc:.4f}, Sen: {best_tmp_test_sen:.4f}, Spe: {best_tmp_test_spe:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}, AUC: {best_val_auc:.4f}, Sen: {best_val_sen:.4f}, Spe: {best_val_spe:.4f}")

    wandb.log({
        'best_test_acc': test_acc,
        'best_test_auc': best_tmp_test_auc,
        'best_test_sen': best_tmp_test_sen,
        'best_test_spe': best_tmp_test_spe,
        'best_val_acc': best_val_acc,
        'best_val_auc': best_val_auc,
        'best_val_sen': best_val_sen,
        'best_val_spe': best_val_spe,
        'best_epoch': best_epoch
    })

    return test_acc, best_val_acc, best_tmp_test_loss, best_tmp_test_auc, best_tmp_test_sen, best_tmp_test_spe, best_epoch


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    model_cls = HyperbolicGraphClassifier

    dataset = get_dataset(args.dataset)

    # Add extra arguments
    # args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # 图分类任务配置
    args.graph_classification = True
    
    if hasattr(args, 'num_classes'):
        args.output_dim = args.num_classes
    else:
        args.output_dim = 2
        args.num_classes = 2
        
    print(f"[INFO] Graph classification mode: {args.num_classes} classes")
    print(f"[INFO] Pool type: {args.pool_type}")
    
    assert args.normalised or args.deg_normalised
    if args.sheaf_decay is None:
        args.sheaf_decay = args.weight_decay

    # Set the seed for everything
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results = []
    wandb.init(project="sheaf", config=vars(args))

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for fold in tqdm(range(args.folds)):
        args_dict = vars(args)
        test_acc, best_val_acc, best_val_loss, test_auc, test_sen, test_spe, best_epoch = run_exp(args_dict, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc, float(best_val_loss), test_auc, test_sen, test_spe])

        # 记录每折结果到日志文件
        log_path = os.path.join(os.path.dirname(__file__), f'experiments_{args.dataset}.log')
        with open(log_path, 'a') as f:
            f.write(f"{datetime.now().isoformat()} model={args.model} dataset={args.dataset} fold={fold} "
                    f"test_acc={test_acc:.4f} test_auc={test_auc:.4f} test_sen={test_sen:.4f} test_spe={test_spe:.4f} "
                    f"best_val_acc={best_val_acc:.4f} best_val_loss={best_val_loss:.6f} "
                    f"best_epoch={best_epoch}\n")

    # 汇总结果
    arr = np.array(results)
    if arr.size == 0:
        print("No results collected.")
    else:
        test_acc_mean = np.mean(arr[:, 0]) * 100
        val_acc_mean = np.mean(arr[:, 1]) * 100
        loss_mean = np.mean(arr[:, 2])
        loss_std = np.std(arr[:, 2])
        test_acc_std = np.std(arr[:, 0]) * 100
        
        test_auc_mean = np.mean(arr[:, 3]) * 100
        test_auc_std = np.std(arr[:, 3]) * 100
        test_sen_mean = np.mean(arr[:, 4]) * 100
        test_sen_std = np.std(arr[:, 4]) * 100
        test_spe_mean = np.mean(arr[:, 5]) * 100
        test_spe_std = np.std(arr[:, 5]) * 100

        wandb.log({
            'test_acc': test_acc_mean,
            'val_acc': val_acc_mean,
            'test_acc_std': test_acc_std,
            'test_auc_mean': test_auc_mean,
            'test_auc_std': test_auc_std,
            'test_sen_mean': test_sen_mean,
            'test_sen_std': test_sen_std,
            'test_spe_mean': test_spe_mean,
            'test_spe_std': test_spe_std,
            'loss_mean': float(loss_mean),
            'loss_std': float(loss_std)
        })
        wandb.finish()

        print(f'{args.model} on {args.dataset}')
        print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
        print(f'Test AUC: {test_auc_mean:.4f} +/- {test_auc_std:.4f}')
        print(f'Test Sen: {test_sen_mean:.4f} +/- {test_sen_std:.4f}')
        print(f'Test Spe: {test_spe_mean:.4f} +/- {test_spe_std:.4f}')
        print(f'Loss mean: {loss_mean:.6f} | Loss std: {loss_std:.6f}')


