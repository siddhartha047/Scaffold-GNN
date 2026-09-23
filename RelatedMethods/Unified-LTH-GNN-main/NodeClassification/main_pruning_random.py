import os
import random
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import net as net
from utils import load_data
import utils
from sklearn.metrics import f1_score
import pdb
import pruning
import copy
from scipy.sparse import coo_matrix
import warnings

SUPPORT_GRAPH_ROOT = Path(os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))
from scripts.common.baseline_result_utils import append_baseline_result
from scaffold_gnn.utils.defaults import DEFAULT_DATA_DIR
warnings.filterwarnings('ignore')

def run_fix_mask(args, seed, adj_percent, wei_percent):

    device = torch.device(args['device'])
    adj, features, labels, idx_train, idx_val, idx_test = load_data(args['dataset'], args['data_root'])
    
    node_num = features.size()[0]
    class_num = labels.numpy().max() + 1

    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn(
        embedding_dim=args['embedding_dim'],
        adj=adj,
        dropout=args['dropout'],
        input_dropout=args['input_dropout'],
        pre_linear=bool(args['pre_linear']),
        residual=bool(args['residual']),
        layer_norm=bool(args['layer_norm']),
        batch_norm=bool(args['batch_norm']),
    )
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.to(device)
    pruning.random_pruning(net_gcn, adj_percent, wei_percent)

    adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
    
    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0, 'train_acc': 0, 'train_f1': 0, 'test_f1': 0}

    for epoch in range(args['total_epoch']):

        optimizer.zero_grad()
        output = net_gcn(features, adj)
        loss = loss_func(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            output = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            acc_train = f1_score(labels[idx_train].cpu().numpy(), output[idx_train].cpu().numpy().argmax(axis=1), average='micro')
            train_f1 = f1_score(labels[idx_train].cpu().numpy(), output[idx_train].cpu().numpy().argmax(axis=1), average='macro', zero_division=0)
            test_f1 = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='macro', zero_division=0)
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['train_acc'] = acc_train
                best_val_acc['train_f1'] = train_f1
                best_val_acc['test_f1'] = test_f1
                best_val_acc['epoch'] = epoch
 
        print("(Fix Mask) Epoch:[{}] Loss:[{:.4f}] Val:[{:.2f}] Test:[{:.2f}] TestF1Macro:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, loss.item(), acc_val * 100, 
                                acc_test * 100, 
                                test_f1 * 100,
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))

    return (
        best_val_acc['val_acc'],
        best_val_acc['test_acc'],
        best_val_acc['epoch'],
        adj_spar,
        wei_spar,
        best_val_acc['train_acc'],
        best_val_acc['train_f1'],
        best_val_acc['test_f1'],
    )


def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    ###### Unify pruning settings #######
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--total_epoch', type=int, default=300)
    parser.add_argument('--pruning_percent', type=float, default=0.1)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
    parser.add_argument('--weight_dir', type=str, default='')
    parser.add_argument('--prune_rounds', type=int, default=20)
    parser.add_argument('--target_kept_ratio', type=float, default=None)
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=None)
    ###### Others settings #######
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--metric', choices=('acc', 'rocauc'), default='acc')
    parser.add_argument('--pre_linear', type=int, choices=(0, 1), default=0)
    parser.add_argument('--residual', type=int, choices=(0, 1), default=0)
    parser.add_argument('--layer_norm', type=int, choices=(0, 1), default=0)
    parser.add_argument('--batch_norm', type=int, choices=(0, 1), default=0)
    return parser


if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)

    seed_dict = {'cora': 3846, 'citeseer': 2839, 'pubmed': 3333}
    seed = args['seed'] if args['seed'] is not None else seed_dict.get(args['dataset'].lower(), 3846)
    if args['embedding_dim'] == [3703, 16, 6]:
        base_dim = utils.infer_embedding_dim(args['data_root'], args['dataset'])
        args['embedding_dim'] = (
            [base_dim[0]]
            + [args['hidden_channels']] * (args['num_layers'] - 1)
            + [base_dim[-1]]
        )

    if args['prune_rounds'] < 1:
        raise ValueError('prune_rounds must be positive')
    requested_kept_ratio = (
        args['target_kept_ratio']
        if args['target_kept_ratio'] is not None
        else (1 - args['pruning_percent_adj']) ** args['prune_rounds']
    )
    percent_list = [
        (
            1 - (1 - args['pruning_percent_adj']) ** (i + 1),
            1 - (1 - args['pruning_percent_wei']) ** (i + 1),
        )
        for i in range(args['prune_rounds'])
    ]
    pruning.setup_seed(seed)
    for run in range(args['runs']):
        os.environ['SCAFFOLD_SPLIT_RUN'] = str(run)
        print(f"[TunedGNNProtocol] run={run + 1}/{args['runs']} seed={seed}")
        final_result = None
        for p, (adj_percent, wei_percent) in enumerate(percent_list):
            final_result = run_fix_mask(
                args, seed, adj_percent, wei_percent
            )
            (
                best_acc_val,
                final_acc_test,
                final_epoch_list,
                adj_spar,
                wei_spar,
                train_acc,
                train_f1,
                test_f1,
            ) = final_result
            print("=" * 120)
            print("syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Final Test F1 Macro:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
                .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, test_f1 * 100, adj_spar, wei_spar))
            print("=" * 120)
            print(
                f"[TargetRatio] requested_kept={requested_kept_ratio:.8f} "
                f"round={p + 1}/{args['prune_rounds']} "
                f"achieved_kept={adj_spar / 100.0:.8f}"
            )
        if final_result is None:
            raise RuntimeError("Unified-LTH produced no pruning round")
        append_baseline_result(
            method='unified_lth',
            dataset=args['dataset'],
            run=run + 1,
            seed=seed,
            epochs=args['total_epoch'],
            kept_ratio=requested_kept_ratio,
            sparsity=100.0 - adj_spar,
            train_acc=100 * train_acc,
            valid_acc=100 * best_acc_val,
            test_acc=100 * final_acc_test,
            train_f1_macro=100 * train_f1,
            test_f1_macro=100 * test_f1,
            chosen_epoch=final_epoch_list,
        )
