#! /usr/bin/env python
# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import random
import torch
import torch.nn.functional as F
# import git
import numpy as np
import wandb
from tqdm import tqdm

# This is required here by wandb sweeps.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from exp.parser import get_parser
from models.positional_encodings import append_top_k_evectors
from models.cont_models import DiagSheafDiffusion, BundleSheafDiffusion, GeneralSheafDiffusion, HyperbolicGraphDiffusion
from models.disc_models import DiscreteDiagSheafDiffusion, DiscreteBundleSheafDiffusion, DiscreteGeneralSheafDiffusion
from utils.heterophilic import get_dataset, get_fixed_splits
from pyG import BrainGraphDataset
from pyG_CInd import BrainGraphCINDDataset
def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x)[data.train_mask]
    nll = F.nll_loss(out, data.y[data.train_mask])
    loss = nll
    loss.backward()

    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x), [], [], []
        # logits 检查
        print("[DEBUG] logits shape:", logits.shape)
        print("[DEBUG] logits contains nan:", torch.isnan(logits).any().item())
        print("[DEBUG] logits contains inf:", torch.isinf(logits).any().item())
        print("[DEBUG] logits max:", logits.max().item(), "| min:", logits.min().item())
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(logits[mask], data.y[mask])
            print(loss)
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses



'''def run_exp(args, dataset, model_cls, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)
    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)
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

        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            break

    print(f"Fold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    if "ODE" not in args['model']:
        # Debugging for discrete models
        for i in range(len(model.sheaf_learners)):
            L_max = model.sheaf_learners[i].L.detach().max().item()
            L_min = model.sheaf_learners[i].L.detach().min().item()
            L_avg = model.sheaf_learners[i].L.detach().mean().item()
            L_abs_avg = model.sheaf_learners[i].L.detach().abs().mean().item()
            print(f"Laplacian {i}: Max: {L_max:.4f}, Min: {L_min:.4f}, Avg: {L_avg:.4f}, Abs avg: {L_abs_avg:.4f}")

        with np.printoptions(precision=3, suppress=True):
            for i in range(0, args['layers']):
                print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_val_acc, keep_running'''
def run_exp(args, dataset, model_cls, fold):
    data = dataset[0]
    data = get_fixed_splits(data, args['dataset'], fold)

    print(f"[DEBUG] Data has {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"[DEBUG] train_mask.sum() = {data.train_mask.sum()}, val_mask.sum() = {data.val_mask.sum()}, test_mask.sum() = {data.test_mask.sum()}")

    data = data.to(args['device'])

    model = model_cls(data.edge_index, args)
    model = model.to(args['device'])

    print(f"[DEBUG] Model structure:\n{model}")

    sheaf_learner_params, other_params = model.grouped_parameters()
    optimizer = torch.optim.Adam([
        {'params': sheaf_learner_params, 'weight_decay': args['sheaf_decay']},
        {'params': other_params, 'weight_decay': args['weight_decay']}
    ], lr=args['lr'])

    epoch = 0
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []
    best_epoch = 0
    bad_counter = 0

    for epoch in range(args['epochs']):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        # Debug acc and loss at each epoch
        print(f"[Epoch {epoch}] train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {tmp_test_acc:.4f}")
        print(f"[Epoch {epoch}] train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, test_loss: {tmp_test_loss:.4f}")

        # Debug predictions
        '''if epoch == 0 or epoch % 10 == 0:
            y_pred = preds.argmax(dim=1).detach().cpu()
            y_true = data.y.detach().cpu()
            print(f"[Epoch {epoch}] Predicted label distribution: {torch.bincount(y_pred)}")
            print(f"[Epoch {epoch}] True label distribution: {torch.bincount(y_true[data.train_mask].cpu())}")'''

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

        new_best_trigger = val_acc > best_val_acc if args['stop_strategy'] == 'acc' else val_loss < best_val_loss
        if new_best_trigger:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args['early_stopping']:
            print(f"[DEBUG] Early stopping at epoch {epoch}")
            break

    print(f"\nFold {fold} | Epochs: {epoch} | Best epoch: {best_epoch}")
    print(f"Test acc: {test_acc:.4f}")
    print(f"Best val acc: {best_val_acc:.4f}")

    if "ODE" not in args['model']:
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

        with np.printoptions(precision=3, suppress=True):
            for i in range(0, args['layers']):
                print(f"Epsilons {i}: {model.epsilons[i].detach().cpu().numpy().flatten()}")

    wandb.log({'best_test_acc': test_acc, 'best_val_acc': best_val_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_val_acc, keep_running



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
    print(f"Running with wandb account: {args.entity}")
    print(args)
    wandb.init(project="sheaf", config=vars(args), entity=args.entity)

    for fold in tqdm(range(args.folds)):
        test_acc, best_val_acc, keep_running = run_exp(wandb.config, dataset, model_cls, fold)
        results.append([test_acc, best_val_acc])
        if not keep_running:
            break

    test_acc_mean, val_acc_mean = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    wandb_results = {'test_acc': test_acc_mean, 'val_acc': val_acc_mean, 'test_acc_std': test_acc_std}
    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model if args.evectors == 0 else f"{args.model}+LP{args.evectors}"
    print(f'{model_name} on {args.dataset} ')
    print(f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Val acc: {val_acc_mean:.4f}')

