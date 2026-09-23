import os
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.utils import remove_self_loops

import net as net
import layers
from args import parser_loader
import utils
from sklearn.metrics import f1_score, roc_auc_score
import pdb
import pruning
import copy
from scipy.sparse import coo_matrix
import warnings
from scripts.common.baseline_result_utils import (
    RunTimeBudget,
    append_baseline_result,
)
warnings.filterwarnings('ignore')


def _selection_metric(labels, logits, indices, metric):
    truth = labels[indices].detach().cpu().numpy()
    selected_logits = logits[indices].detach()
    if metric == "rocauc":
        probabilities = torch.softmax(selected_logits, dim=-1).cpu().numpy()
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(truth, probabilities[:, 1]))
        return float(
            roc_auc_score(
                truth,
                probabilities,
                multi_class="ovr",
            )
        )
    prediction = selected_logits.argmax(dim=1).cpu().numpy()
    return float(f1_score(truth, prediction, average="micro"))


def run_get_mask(args, budget=None, run_number=1):
    device = args['device']
    
    adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
        utils.load_citation(args['dataset'], task_type=args['task_type'])  # adj: csr_matrix

    adj = adj.to_dense().to(device).to(torch.float32)
    adj = adj.nonzero().t().contiguous()
    features = features.to(device).to(torch.float32)
    labels = labels.to(device)
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn_dense(
        embedding_dim=args['embedding_dim'],
        out_channels=args['out_channels'],
        edge_index=adj,
        device=device,
        spar_wei=False,
        spar_adj=args['spar_adj'],
        num_nodes=features.shape[0],
        use_bn=args['use_bn'],
        use_res=args['use_res'],
        use_ln=args['use_ln'],
        dropout=args['dropout'],
        input_dropout=args['input_dropout'],
        pre_linear=bool(args['pre_linear']),
        jumping_knowledge=bool(args['jumping_knowledge']),
        coef=args['coef'],
    )
    net_gcn = net_gcn.to(device)

    optimizer = torch.optim.Adam(
        [
            {
                "params": list(net_gcn.backbone_parameters()),
                "weight_decay": args['weight_decay'],
            },
            {
                "params": list(net_gcn.sparsifier_parameters()),
                # tunedGNN weight decay belongs to the GCN, not the auxiliary
                # edge scores/thresholds used to discover the sparse graph.
                "weight_decay": 0.0,
            },
        ],
        lr=args['lr'],
    )

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "adj_spar": 0, "wei_spar": 0}
    best_target = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, "mask": None}
    best_mask = None
    adj_mask_ls, wei_mask_ls = [], []
    
    rewind_weight = copy.deepcopy(net_gcn.state_dict())
    for epoch in range(args['total_epoch']):        
        net_gcn.train()

        optimizer.zero_grad()
        output = net_gcn(features, adj, pretrain=(epoch < args['pretrain_epoch']))

        loss = loss_func(output[idx_train], labels[idx_train])

        adj_loss = 0
        if args['spar_adj']:
            for thres in net_gcn.adj_thresholds:
                adj_loss += torch.exp(-thres).sum() / args['num_layers']
            adj_loss = (adj_loss * args['e1'])
            loss = loss + adj_loss

        wei_loss = 0
        if args['spar_wei']:
            for layer in net_gcn.modules():
                if isinstance(layer, layers.MaskedLinear):
                    wei_loss += args['e2'] * \
                        torch.sum(torch.exp(-layer.threshold))
        # print(wei_loss)
        loss += wei_loss

        loss.backward()
        
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(features, adj, val_test=True, pretrain=(epoch < args['pretrain_epoch']))
            acc_val = _selection_metric(
                labels, output, idx_val, args['metric']
            )
            acc_test = _selection_metric(
                labels, output, idx_test, args['metric']
            )
            acc_train = _selection_metric(
                labels, output, idx_train, args['metric']
            )

            aspar_here = utils.calcu_sparsity(net_gcn.edge_mask_archive, adj.shape[1])
            wspar_here = utils.net_weight_sparsity(net_gcn)

            # continuous setting for adj/wei
            in_interval = (args['spar_adj'] and utils.judge_spar(aspar_here, args['target_adj_spar']) or not args['spar_adj']) and \
                (args['spar_wei'] and utils.judge_spar(wspar_here, args['target_wei_spar']) or not args['spar_wei'])
            if in_interval and args['continuous']:
                # print(net_gcn.edge_mask_archive)
                adj_mask_ls.append(copy.deepcopy(net_gcn.edge_mask_archive))
                wei_mask_ls.append(copy.deepcopy(net_gcn.generate_wei_mask()))
        
            # for report
            meet = ((args['spar_adj'] and aspar_here > args['target_adj_spar']) or not args['spar_adj']) and \
                    ((args['spar_wei'] and wspar_here > args['target_wei_spar']) or not args['spar_wei']) 
            if acc_val > best_val_acc['val_acc'] and meet:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_mask = [copy.deepcopy(net_gcn.edge_mask_archive), copy.deepcopy(net_gcn.generate_wei_mask())]
                best_val_acc['wei_spar'] = wspar_here
                best_val_acc['adj_spar'] = aspar_here

            # sole setting for adj/wei
            if in_interval and acc_val > best_target['val_acc']:
                best_target['test_acc'] = acc_test
                best_target['val_acc'] = acc_val
                best_target['epoch'] = epoch
                best_target['mask'] = [copy.deepcopy(net_gcn.edge_mask_archive), copy.deepcopy(net_gcn.generate_wei_mask())]

            print("Epoch:[{}] L:[{:.3f}] AL:[{:.2f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] WS:[{:.2f}%] AS:[{:.2f}%] |"
                  .format(epoch, loss.item(), (adj_loss), acc_train * 100, acc_val * 100, acc_test * 100, wspar_here, aspar_here), end=" ")
            if meet:
                print("Best Val:[{:.2f}] Test:[{:.2f}] AS:[{:.2f}%] WS:[{:.2f}%] at Epoch:[{}]"
                      .format(
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['test_acc'] * 100,
                          best_val_acc['adj_spar'],
                          best_val_acc['wei_spar'],
                          best_val_acc['epoch']))
            else:
                print("")

        # Stop only after a complete mask-learning epoch.  The selected mask
        # below is still projected to the requested edge budget, and the fixed
        # ticket phase gets one completed train/evaluation epoch so a valid
        # result is always emitted at the wall-clock deadline.
        if budget is not None and budget.exhausted(
            run_number,
            f"mask:{epoch + 1}",
            f"mask:{args['total_epoch']}+fixed:{args['retain_epoch']}",
        ):
            break

    if best_target['mask'] is None:
        print("Target sparsity was not reached; using the latest masks for smoke/short-run compatibility.")
        best_target['mask'] = [copy.deepcopy(net_gcn.edge_mask_archive), copy.deepcopy(net_gcn.generate_wei_mask())]

    if args['continuous']:
        return adj_mask_ls, wei_mask_ls, None, rewind_weight
    selected_edge_masks, selected_weight_masks = best_target['mask']
    requested_kept_ratio = 1.0 - args['target_adj_spar'] / 100.0
    selected_edge_masks = utils.enforce_edge_keep_ratio(
        selected_edge_masks,
        adj,
        requested_kept_ratio,
    )
    actual_adj_sparsity = utils.calcu_sparsity(
        selected_edge_masks,
        adj.shape[1],
    )
    print(
        f"[TargetRatio] requested_kept={requested_kept_ratio:.8f} "
        f"achieved_kept={1.0 - actual_adj_sparsity / 100.0:.8f} "
        f"achieved_sparsity={actual_adj_sparsity:.6f}%"
    )
    return selected_edge_masks, selected_weight_masks, actual_adj_sparsity, rewind_weight


def run_fix_mask(
    args,
    edge_masks,
    wei_masks,
    rewind_weight=None,
    budget=None,
    run_number=1,
):
    device = args['device']

    edge_masks = [mask.to(device) for mask in edge_masks]
    wei_masks = [mask.to(device) for mask in wei_masks]
    
    adj, features, labels, idx_train, idx_val, idx_test, degree, learning_type = \
        utils.load_citation(args['dataset'])  # adj: csr_matrix

    adj = adj.to_dense().to(device).to(torch.float32)
    adj = adj.nonzero().t().contiguous()
    features = features.to(device).to(torch.float32)
    labels = labels.to(device)
    loss_func = nn.CrossEntropyLoss()
    
    net_gcn = net.net_gcn_dense(
        embedding_dim=args['embedding_dim'],
        out_channels=args['out_channels'],
        edge_index=adj,
        device=device,
        spar_wei=False,
        spar_adj=False,
        num_nodes=features.shape[0],
        use_bn=args['use_bn'],
        use_res=args['use_res'],
        use_ln=args['use_ln'],
        dropout=args['dropout'],
        input_dropout=args['input_dropout'],
        pre_linear=bool(args['pre_linear']),
        jumping_knowledge=bool(args['jumping_knowledge']),
        mode="retain",
    )
    net_gcn = net_gcn.to(device)

    utils.rewind_compatible_weights(net_gcn, rewind_weight)

    optimizer = torch.optim.Adam(net_gcn.parameters(
    ), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch': 0, 'test_acc': 0, 'train_acc': 0, 'train_f1': 0, 'test_f1': 0}
    full_graph_eval = os.environ.get(
        "BASELINE_EVAL_GRAPH", "sparse"
    ).strip().lower() == "original"
    evaluation_edge_masks = None if full_graph_eval else edge_masks
    if full_graph_eval:
        print(
            "[EvaluationGraph] topology=original-full "
            "training_topology=fixed-sparse-ticket",
            flush=True,
        )
    
    for epoch in range(args['retain_epoch']):
        net_gcn.train()
        optimizer.zero_grad()
        output = net_gcn(features, adj, edge_masks=edge_masks,wei_masks=wei_masks)

        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        
        optimizer.step()
        with torch.no_grad():
            net_gcn.eval()
            output = net_gcn(features, adj, val_test=True,
                             edge_masks=evaluation_edge_masks,
                             wei_masks=wei_masks)
            acc_val = _selection_metric(
                labels, output, idx_val, args['metric']
            )
            acc_test = _selection_metric(
                labels, output, idx_test, args['metric']
            )
            acc_train = _selection_metric(
                labels, output, idx_train, args['metric']
            )
            train_f1 = f1_score(labels[idx_train].cpu().numpy(),
                                output[idx_train].cpu().numpy().argmax(axis=1),
                                average='macro', zero_division=0)
            test_f1 = f1_score(labels[idx_test].cpu().numpy(),
                               output[idx_test].cpu().numpy().argmax(axis=1),
                               average='macro', zero_division=0)
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['train_acc'] = acc_train
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_val_acc['train_f1'] = train_f1
                best_val_acc['test_f1'] = test_f1

            print("Epoch:[{}] Loss:[{:.4f}] Train:[{:.2f}] Val:[{:.2f}] Test:[{:.2f}] TestF1Macro:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                  .format(epoch, loss.item(), acc_train * 100 ,acc_val * 100, acc_test * 100,
                          test_f1 * 100,
                          best_val_acc['val_acc'] * 100,
                          best_val_acc['test_acc'] * 100,
                          best_val_acc['epoch']))
        if budget is not None and budget.exhausted(
            run_number,
            f"fixed:{epoch + 1}",
            f"mask:{args['total_epoch']}+fixed:{args['retain_epoch']}",
        ):
            break
    return best_val_acc

if __name__ == "__main__":

    args = parser_loader()
    print(args)
    print("[PruningMode] edge_only=true weight_masks=disabled")
    # torch.autograd.set_detect_anomaly#(True)
    utils.fix_seed(args['seed'])
    
    for run in range(args['runs']):
        os.environ['SCAFFOLD_SPLIT_RUN'] = str(run)
        print(f"[TunedGNNProtocol] run={run + 1}/{args['runs']} seed={args['seed']}")
        budget = RunTimeBudget().start()
        edge_masks, wei_masks, actual_adj_sparsity, rewind_weight = run_get_mask(
            args,
            budget=budget,
            run_number=run + 1,
        )

        if not args['continuous']:
            best = run_fix_mask(
                args,
                edge_masks,
                wei_masks,
                rewind_weight=rewind_weight,
                budget=budget,
                run_number=run + 1,
            )
            append_baseline_result(
                method='adaglt',
                dataset=args['dataset'],
                run=run + 1,
                seed=args['seed'],
                epochs=args['retain_epoch'],
                kept_ratio=1.0 - float(args['target_adj_spar']) / 100.0,
                sparsity=actual_adj_sparsity,
                train_acc=100 * best['train_acc'],
                valid_acc=100 * best['val_acc'],
                test_acc=100 * best['test_acc'],
                train_f1_macro=100 * best['train_f1'],
                test_f1_macro=100 * best['test_f1'],
                chosen_epoch=best['epoch'],
            )
        else:
            print(len(edge_masks))
            for emask, wmask in zip(edge_masks, wei_masks):
                run_fix_mask(
                    args, emask, wmask, rewind_weight=rewind_weight
                )
