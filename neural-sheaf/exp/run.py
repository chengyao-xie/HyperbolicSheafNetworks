#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
# import git
import numpy as np
import wandb
from tqdm import tqdm
from datetime import datetime

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion, HyperbolicGraphDiffusion, HyperbolicGraphClassifier
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from utils.heterophilic import get_dataset, get_fixed_splits
from utils.pyG import BrainGraphDataset

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

# 统一的标签提取函数
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

# 为对比学习准备单个图数据
def prepare_graph_data_for_contrastive(graph_data_item, device, add_noise=False, noise_scale=0.01):
    graph_data_item = graph_data_item.to(device)
    
    # 添加数据增强噪声
    if add_noise:
        noise = torch.randn_like(graph_data_item.x) * noise_scale
        graph_data_item.x = graph_data_item.x + noise
    
    # 提取标签
    labels = extract_graph_labels(graph_data_item)
    
    return graph_data_item.x, {'y': labels}

# 批量对比学习训练（在函数内部完成反传与更新，返回标量损失）
def batch_contrastive_training(model, batch_graphs, device, epoch, optimizer, grad_clip=1.0, scaler: GradScaler = None):
    embeddings_list = []
    data_list = []
    
    for graph_data_item in batch_graphs:
        embeddings, data_dict = prepare_graph_data_for_contrastive(
            graph_data_item, device, add_noise=(epoch > 10), noise_scale=0.01
        )
        embeddings_list.append(embeddings)
        data_list.append(data_dict)
    
    # 计算对比学习损失
    use_amp = (scaler is not None) and scaler.is_enabled()
    with autocast(enabled=use_amp):
        metrics = model.compute_metrics_multiple(embeddings_list, data_list, 'train')
        loss = metrics['loss']
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
    # 仅返回标量用于日志
    return loss.detach()

# 批量标准训练（按图逐个微批次训练，单图前向/反传/更新，返回该batch平均标量损失）
def batch_standard_training(model, batch_graphs, device, epoch, optimizer, grad_clip=1.0, scaler: GradScaler = None):
    batch_loss = 0.0
    for graph_data_item in batch_graphs:
        graph_data_item = graph_data_item.to(device)
        
        # 添加数据增强噪声
        if epoch > 10:
            noise = torch.randn_like(graph_data_item.x) * 0.01
            graph_data_item.x = graph_data_item.x + noise
        
        use_amp = (scaler is not None) and scaler.is_enabled()
        with autocast(enabled=use_amp):
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
        # 可选：释放碎片显存
        # torch.cuda.empty_cache()
    return batch_loss / max(1, len(batch_graphs))

# 对比学习评估
def evaluate_contrastive(model, data_samples, device):
    if len(data_samples) >= 2 and model.use_contrastive:
        # 准备数据用于对比学习评估
        embeddings_list = []
        data_list = []
        
        for graph_data_item in data_samples:
            embeddings, data_dict = prepare_graph_data_for_contrastive(graph_data_item, device)
            embeddings_list.append(embeddings)
            data_list.append(data_dict)
        
        # 计算对比学习指标
        metrics = model.compute_metrics_multiple(embeddings_list, data_list, 'eval')
        return metrics['acc'], metrics['loss'].item()
    else:
        # 回退到标准评估
        acc_sum, loss_sum = 0, 0
        for graph_data_item in data_samples:
            graph_data_item = graph_data_item.to(device)
            acc, loss, _ = test_graph_classification(model, graph_data_item)
            acc_sum += acc
            loss_sum += loss
        return acc_sum / len(data_samples), loss_sum / len(data_samples)

# 标准化标签处理
def normalize_labels(labels, device):
    if isinstance(labels, (int, float)):
        return torch.tensor(labels, dtype=torch.float32, device=device)
    elif labels.dim() > 0 and labels.numel() == 1:
        return labels.float().squeeze()
    else:
        return labels.float()

# 标准化输出处理
def normalize_logits(logits):
    if logits.dim() > 0:
        if logits.numel() == 1:
            return logits.squeeze()
        elif logits.size(-1) == 1:
            logits = logits.squeeze(-1)
            if logits.dim() > 0 and logits.numel() == 1:
                return logits.squeeze()
    return logits

# 计算图分类损失
def compute_graph_classification_loss(model, data):
    labels = extract_graph_labels(data)
    labels = normalize_labels(labels, data.x.device)
    
    out = model(data.x)
    out = normalize_logits(out)
    
    # 获取分类数量
    num_classes = getattr(model, 'num_classes', 
                         getattr(getattr(model, 'graph_diffusion', None), 'output_dim', 2))
    
    if num_classes == 2:
        return F.binary_cross_entropy_with_logits(out, labels)
    else:
        if out.dim() == 0:
            out = out.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        return F.cross_entropy(out, labels.long())

#简化的图分类测试函数
def test_graph_classification(model, data):
    model.eval()
    with torch.no_grad():
        labels = extract_graph_labels(data)
        labels = normalize_labels(labels, data.x.device)
        
        logits = model(data.x)
        logits = normalize_logits(logits)
        
        # 获取分类数量
        num_classes = getattr(model, 'num_classes',
                             getattr(getattr(model, 'graph_diffusion', None), 'output_dim', 2))
        
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
            pred = torch.argmax(logits, dim=-1)
        
        acc = (pred == labels).float().mean().item()
        return acc, loss.item(), pred.detach().cpu()

# 原始节点分类训练函数
def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out

# 原始节点分类测试函数
def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        # logits 检查
        # print("[DEBUG] logits shape:", logits.shape)
        # print("[DEBUG] logits contains nan:", torch.isnan(logits).any().item())
        # print("[DEBUG] logits contains inf:", torch.isinf(logits).any().item())
        # print("[DEBUG] logits max:", logits.max().item(), "| min:", logits.min().item())
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])
            # print(loss)
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses


# 评估图分类模型
def evaluate_graph_classification(model, data_list, device):
    model.eval()
    total_correct = 0
    total_loss = 0
    total_samples = len(data_list)
    
    if total_samples == 0:
        return 0.0, float('inf')
    
    with torch.no_grad():
        for graph_data in data_list:
            graph_data = graph_data.to(device)
            acc, loss, pred = test_graph_classification(model, graph_data)
            total_correct += acc  # acc已经是准确率(0-1之间)
            total_loss += loss
    
    avg_acc = total_correct / total_samples
    avg_loss = total_loss / total_samples
    
    return avg_acc, avg_loss


def init_metric_dict():
    return {'acc': -1, 'roc': -1, 'ap': -1, 'loss': float('inf')}

# 检查是否有改进
def has_improved(m1, m2):
    return m1['acc'] > m2['acc']

# 重置每个epoch的统计信息
def reset_epoch_stats(self, epoch, prefix):
    if (prefix != 'start') and hasattr(self, 'epoch_stats') and ('start' != self.epoch_stats['prefix']):
        if self.epoch_stats['prefix'] not in self.metrics_tracker:
            self.metrics_tracker[self.epoch_stats['prefix']] = []
        avg_stats, _ = self.report_epoch_stats()
        self.metrics_tracker[self.epoch_stats['prefix']].append(avg_stats)
        
    self.epoch_stats = {
        'prefix': prefix,
        'epoch': epoch,
        'loss': 0,
        'acc': 0,
        'roc': 0,
        'ap': 0,
        'num_graphs': 0,
        'num_updates': 0
    }

# 更新epoch统计信息
def update_epoch_stats(self, metrics, split):
    with torch.no_grad():
        self.epoch_stats['loss'] += metrics['loss'].item()
        self.epoch_stats['acc'] += metrics['acc']
        self.epoch_stats['roc'] += metrics['roc']
        self.epoch_stats['ap'] += metrics['ap']
        self.epoch_stats['num_graphs'] += metrics['num_graphs']
        self.epoch_stats['num_updates'] += 1

# 报告epoch统计信息
def report_epoch_stats(self):
    if self.epoch_stats['num_updates'] == 0:
        return {}, ""
        
    avg_loss = self.epoch_stats['loss'] / self.epoch_stats['num_updates']
    avg_acc = self.epoch_stats['acc'] / self.epoch_stats['num_updates']
    avg_roc = self.epoch_stats['roc'] / self.epoch_stats['num_updates']
    avg_ap = self.epoch_stats['ap'] / self.epoch_stats['num_updates']
    
    avg_stats = {
        'prefix': self.epoch_stats['prefix'],
        'epoch': self.epoch_stats['epoch'],
        'loss': avg_loss,
        'acc': avg_acc,
        'roc': avg_roc,
        'ap': avg_ap
    }
    
    stat_string = "%s phase of epoch %d: loss %.6f, acc %.6f, roc %.6f, ap %.6f, #graphs %d" % (
        self.epoch_stats['prefix'],
        self.epoch_stats['epoch'],
        avg_loss,
        avg_acc,
        avg_roc,
        avg_ap,
        self.epoch_stats['num_graphs']
    )
    
    return avg_stats, stat_string


def run_exp(args, dataset, model_cls, fold):
    # 检查是否为图分类任务
    is_graph_classification = args.get('task', 'node_classification') == 'graph_classification'
    
    if is_graph_classification:
        # 图分类：需要多个图进行训练/验证/测试划分
        if len(dataset) < 10:
            print(f"Warning: Dataset only has {len(dataset)} graphs, which is too small for proper train/val/test split")
            print("Using simple split for demonstration...")
        
        # 创建图级别的数据划分
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
        
        # 创建数据加载器或直接使用图列表
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
        
        # 为模型初始化使用第一个图的结构
        sample_data = dataset[0].to(args['device'])
    else:
        # 节点分类：使用固定分割
        data = dataset[0]
        data = get_fixed_splits(data, args['dataset'], fold)
        data = data.to(args['device'])
        sample_data = data

    model = model_cls(sample_data.edge_index, args)
    model = model.to(args['device'])
    
    # 打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable: {trainable_params}")
    
    # AMP scaler
    use_amp = bool(args.get('use_amp', False) and torch.cuda.is_available())
    scaler = GradScaler(enabled=use_amp)

    # 将外部函数绑定到模型实例
    model.init_metric_dict = init_metric_dict
    model.has_improved = has_improved
    model.reset_epoch_stats = reset_epoch_stats.__get__(model)
    model.update_epoch_stats = update_epoch_stats.__get__(model)
    model.report_epoch_stats = report_epoch_stats.__get__(model)

    # 使用更保守的学习率调度器和优化器设置
    sheaf_learner_params, other_params = model.grouped_parameters()
    
    # 为图分类任务调整学习率
    base_lr = args['lr'] * 0.5 if is_graph_classification else args['lr']
    
    optimizer = torch.optim.AdamW([  # 使用AdamW优化器
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay'], 'lr': base_lr},
        {'params': other_params, 'weight_decay': args['weight_decay'], 'lr': base_lr}
    ], betas=(0.9, 0.999), eps=1e-8)
    
    # 使用StepLR调度器，更平滑的学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args['epochs'] // 4,  # 每1/4个epoch衰减一次
        gamma=0.8  # 每次衰减到80%
    )
    
    epoch = 0
    best_val_acc = test_acc = best_tmp_test_acc = 0
    best_val_loss = float('inf')
    best_tmp_test_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0
    
    # 添加学习率衰减和更灵活的早停策略
    patience = args.get('early_stopping', 50)  # 增加patience
    min_delta = 0.001  # 最小改进阈值

    for epoch in range(args['epochs']):
        # 学习率预热
        if epoch < args['warm_start']:
            lr_scale = min(1., float(epoch + 1) / float(args['warm_start']))
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * args['lr']
        
        if is_graph_classification:
            model.train()
            running_loss = 0.0
            max_train_graphs = min(50, len(train_data))
            train_subset = random.sample(train_data, max_train_graphs)
            batch_size = min(8, len(train_subset))
            num_batches = 0
             
            for i in range(0, len(train_subset), batch_size):
                batch_graphs = train_subset[i:i + batch_size]
                # 选择训练方式
                if len(batch_graphs) >= 2 and model.use_contrastive:
                    batch_loss = batch_contrastive_training(model, batch_graphs, args['device'], epoch, optimizer, args['grad_clip'], scaler)
                else:
                    batch_loss = batch_standard_training(model, batch_graphs, args['device'], epoch, optimizer, args['grad_clip'], scaler)
                running_loss += float(batch_loss.item() if torch.is_tensor(batch_loss) else batch_loss)
                num_batches += 1
            avg_loss = running_loss / max(1, num_batches)
            
            # 更频繁的评估 - 每个epoch都评估
            train_sample = train_subset[:min(16, len(train_subset))]
            val_sample = val_data[:min(16, len(val_data))]
            test_sample = test_data[:min(16, len(test_data))]
            
            train_acc, train_loss = evaluate_contrastive(model, train_sample, args['device'])
            val_acc, val_loss = evaluate_contrastive(model, val_sample, args['device'])
            tmp_test_acc, tmp_test_loss = evaluate_contrastive(model, test_sample, args['device'])
            
            # 打印详细的训练信息
            learning_type = "Contrastive Learning" if model.use_contrastive else "Standard"
            print(f"[Epoch {epoch}] {learning_type} Training Loss: {avg_loss:.4f}")
            
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
        else:
            # 节点分类训练（可选使用 AMP）
            model.train()
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                out = model(data.x)[data.train_mask]
                loss = F.nll_loss(out, data.y[data.train_mask])
            # 添加标签平滑正则化
            eps = 0.1
            n_classes = out.size(1)
            log_preds = F.log_softmax(out, dim=-1)
            loss = (1 - eps) * loss + eps * (-log_preds.mean() / n_classes)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)
        
        scheduler.step()
        
        # Debug acc and loss at each epoch
        print(f"[Epoch {epoch}] train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {tmp_test_acc:.4f}")
        print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, test_loss: {tmp_test_loss:.4f}")

        if fold == 0:
            res_dict = {
                f'fold{fold}_train_acc': train_acc,
                f'fold{fold}_train_loss': train_loss,
                f'fold{fold}_val_acc': val_acc,
                f'fold{fold}_val_loss': val_loss,
                f'fold{fold}_tmp_test_acc': tmp_test_acc,
                f'fold{fold}_tmp_test_loss': tmp_test_loss,
            }
            wandb.log(res_dict, step=epoch)

        new_best_trigger = val_acc > best_val_acc + min_delta if args['stop_strategy'] == 'acc' else val_loss < best_val_loss - min_delta
        if new_best_trigger:
            best_tmp_test_acc = tmp_test_acc
            best_tmp_test_loss = tmp_test_loss
            test_acc = tmp_test_acc
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        # 动态调整学习率
        if bad_counter > 0 and bad_counter % 15 == 0:  # 每15个无改进epoch调整一次
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5  # 将学习率减半
            print(f"[INFO] Reducing learning rate to {param_group['lr']:.6f}")

        if bad_counter >= patience:
            print(f"[DEBUG] Early stopping at epoch {epoch}")
            break

    print(f"\nFold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    if "ODE" not in args['model']:
        # 检查模型是否有 sheaf_learners 属性
        if hasattr(model, 'sheaf_learners') and len(model.sheaf_learners) > 0:
            for i in range(len(model.sheaf_learners)):
                L = model.sheaf_learners[i].L
                if L is not None:
                    L_max = L.detach().max().item()
                    L_min = L.detach().min().item()
                    L_avg = L.detach().mean().item()
                    L_abs_avg = L.detach().abs().mean().item()
                    print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")
                else:
                    print(f"[WARNING] Laplacian {i} is None")

        # 检查模型是否有 epsilons 属性
        if hasattr(model, 'epsilons') and len(model.epsilons) > 0:
            with np.printoptions(precision=3, suppress=True):
                for i in range(0, args['layers']):
                    if i < len(model.epsilons):
                        print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_val_acc, keep_running, best_tmp_test_loss, best_epoch



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # repo = git.Repo(search_parent_directories=True)
    # sha = repo.head.object.hexsha

    if args.model == 'DiagSheafODE':
        model_cls = DiagSheafDiffusion
    elif args.model == 'BundleSheafODE':
        model_cls = BundleSheafDiffusion
    elif args.model == 'GeneralSheafODE':
        model_cls = GeneralSheafDiffusion
    elif args.model == 'DiagSheaf':
        model_cls = DiscreteDiagSheafDiffusion
    elif args.model == 'BundleSheaf':
        model_cls = DiscreteBundleSheafDiffusion
    elif args.model == 'GeneralSheaf':
        model_cls = DiscreteGeneralSheafDiffusion
    elif args.model == 'HyperbolicSheafODE':
        # 检查是否为图分类任务
        if hasattr(args, 'task') and args.task == 'graph_classification':
            model_cls = HyperbolicGraphClassifier
        else:
            model_cls = HyperbolicGraphDiffusion
    else:
        raise ValueError(f'Unknown model {args.model}')

    dataset = get_dataset(args.dataset)
    if args.evectors > 0:
        dataset = append_top_k_evectors(dataset, args.evectors)

    # Add extra arguments
    # args.sha = sha
    args.graph_size = dataset[0].x.size(0)
    args.input_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    
    # 添加图分类任务支持
    if hasattr(args, 'task') and args.task == 'graph_classification':
        # 图分类任务的特殊处理
        args.graph_classification = True
        # 确保有正确的图级别标签数量
        if hasattr(args, 'num_classes'):
            args.output_dim = args.num_classes
        else:
            # 如果没有明确指定，默认为二分类
            args.output_dim = 2
            args.num_classes = 2
            
        # 对比学习参数
        if not hasattr(args, 'use_contrastive'):
            args.use_contrastive = True  # 默认启用对比学习
        if not hasattr(args, 'contrastive_weight'):
            args.contrastive_weight = 0.3  # 对比学习权重
        if not hasattr(args, 'temperature'):
            args.temperature = 0.1  # 对比学习温度
        if not hasattr(args, 'projection_dim'):
            args.projection_dim = 64  # 投影维度
            
        print(f"[INFO] Graph classification mode: {args.num_classes} classes")
        print(f"[INFO] Contrastive Learning: {'Enabled' if args.use_contrastive else 'Disabled'}")
        if args.use_contrastive:
            print(f"[INFO] Contrastive weight: {args.contrastive_weight}, Temperature: {args.temperature}")
    else:
        args.graph_classification = False
        args.task = 'node_classification'
    
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
    # print(f"Running with wandb account: {args.entity}")
    # print(args)
    wandb.init(project="sheaf", config=vars(args), entity=args.entity)

    # 启用 CUDNN benchmark 以提速
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    for fold in tqdm(range(args.folds)):
        # 使用原始 args 
        args_dict = vars(args)
        test_acc, best_val_acc, keep_running, best_val_loss, best_epoch = run_exp(args_dict, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc, float(best_val_loss)])

        # Append per-fold summary to a log file
        log_path = os.path.join(os.path.dirname(__file__), f'experiments_{args.dataset}.log')
        with open(log_path, 'a') as f:
            f.write(f"{datetime.now().isoformat()} model={args.model} dataset={args.dataset} fold={fold} test_acc={test_acc:.4f} best_val_acc={best_val_acc:.4f} best_val_loss={best_val_loss:.6f} best_epoch={best_epoch}\n")

        if not keep_running:
            break
    # Aggregate results (results: [test_acc, val_acc, val_loss] per run)
    arr = np.array(results)
    if arr.size == 0:
        print("No results collected.")
    else:
        test_acc_mean = np.mean(arr[:, 0]) * 100
        val_acc_mean = np.mean(arr[:, 1]) * 100
        loss_mean = np.mean(arr[:, 2])
        loss_std = np.std(arr[:, 2])
        test_acc_std = np.std(arr[:, 0]) * 100

        wandb_results = {
            'test_acc': test_acc_mean,
            'val_acc': val_acc_mean,
            'test_acc_std': test_acc_std,
            'loss_mean': float(loss_mean),
            'loss_std': float(loss_std)
        }
        wandb.log(wandb_results)
        wandb.finish()

        model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
        print(f'{model_name} on {args.dataset} ')
        print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')
        print(f'Loss mean: {loss_mean:.6f} | Loss std: {loss_std:.6f}')


