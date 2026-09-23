import os
import os.path as osp
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import Compose, ToSparseTensor, ToUndirected, AddSelfLoops, NormalizeFeatures
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
import time

SUPPORT_GRAPH_ROOT = Path(os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))
from BenchmarkDataset import load_pyg_data, select_pyg_split
from scripts.common.baseline_result_utils import (
    RunTimeBudget,
    macro_f1_percent,
    single_label_metric_percent,
)

import numpy as np
import random

from logger import Logger
from args import parser_loader
from MoG import MoG

from contextlib import contextmanager
import subprocess
import re

def get_gpu_memory_usage():
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        return [int(x) for x in result.decode('utf-8').strip().split('\n')]
    except Exception:
        return []

# Call this periodically during training to track memory usage
#get_gpu_memory_usage()


def train(model:MoG,features,indices,labels,values,shape,train_idx,temp,optimizer,mask=None):
    """
    :params model: MoG
    :params values: value of edge(all one)
    :params shape: shape of adj matrix
    :parsms temp: temp of sparsification learner
    
    :return: loss
    """
    model.train()
    optimizer.zero_grad()
    mask,add_loss = model.learner(x = features, edge_index = indices, 
                                  temp = temp,shape = shape,
                                  edge_attr = values, training = True)  # masks:size(num_edges)
    output = model.gnn(features, indices, mask)
    supervised_loss = F.nll_loss(output[train_idx], labels[train_idx])
    # The MoE load-balancing term is a loss, not a logit offset.  Adding it to
    # every log-probability before NLL reverses its gradient because NLL
    # negates the selected value.
    loss = supervised_loss + add_loss
    loss.backward()   
    optimizer.step()
    return loss.item(),mask
    

@torch.no_grad()
def test(model:MoG,features,indices,labels,values,shape,split_idx,temp,mask=None,metric='acc'):
    model.eval()
    mask,add_loss = model.learner(x = features, edge_index = indices,
                                  temp = temp,shape = shape,
                                  edge_attr = values, training = False)  # mask:size(num_edges)
    evaluation_mask = mask
    if os.environ.get('BASELINE_EVAL_GRAPH', 'sparse').lower() == 'original':
        evaluation_mask = torch.ones_like(mask)
    output = model.gnn(features, indices, evaluation_mask)
    sparsity = torch.nonzero(mask).size(0)/mask.numel()
    y_pred = output.argmax(dim=-1, keepdim=False)
    # Minesweeper and Questions are scored by ROC-AUC, so the reported value and
    # the best-epoch selection in main() both have to follow --metric.  Their
    # accuracy is the 80%/97% majority-class rate whatever the model learns.
    if str(metric).lower() == 'rocauc':
        def primary(split):
            idx = split_idx[split]
            return single_label_metric_percent(labels[idx], output[idx], metric) / 100.0
    else:
        def primary(split):
            idx = split_idx[split]
            return y_pred[idx].eq(labels[idx]).sum().item()/idx.sum().item()

    train_acc = primary('train')
    valid_acc = primary('valid')
    test_acc = primary('test')
    train_f1 = macro_f1_percent(labels[split_idx['train']], y_pred[split_idx['train']]) / 100.0
    test_f1 = macro_f1_percent(labels[split_idx['test']], y_pred[split_idx['test']]) / 100.0

    return train_acc, valid_acc, test_acc, sparsity, train_f1, test_f1
    

def save_edge_index(edge_index,mask,sparsity):
    edge_index = edge_index[:,mask.bool()]
    torch.save({'edge_index':edge_index},f'edge_index/{sparsity:.2f}_edge_index.pt')

def main():
    args = parser_loader()
    print(args)
    if args['device'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args['device']
    fix_seed(args['seed'])
    device = f"cuda:{args['device']}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset_name = args['dataset']
    
    # dataset = Planetoid(root='./results/data/Planetoid', name=dataset_name, transform=NormalizeFeatures())
    # path= osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', f'{dataset_name}.pt')
    # dataset = torch.load(path)
    
    #data = dataset[0]
    #print(data)

    data, dataset = load_pyg_data(args['data_root'], dataset_name)

    if len(data.train_mask.shape)>1:
        index = 0
        data.train_mask = data.train_mask[:,index]
        data.val_mask = data.val_mask[:,index]
        if dataset_name == 'wiki':
            None
        else:
            data.test_mask = data.test_mask[:,index]

    # num_edges = data.edge_index.shape[1]
    # sampled_indices = torch.randperm(num_edges)[:int(num_edges/2)]
    # sampled_edge_index = data.edge_index[:, sampled_indices]
    # data.edge_index = sampled_edge_index

    #data.adj_t = 
    #data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    split_idx = {'train':data.train_mask,'valid':data.val_mask,'test':data.test_mask}
    train_idx = split_idx['train'].to(device)

    # transform the data
    features = data.x
    labels = data.y

    print(min(labels),max(labels))

    num_classes = max(labels).item() + 1

    #row,col,_= data.adj_t.coo()
    #indices = torch.stack([row,col],dim=0)

    # Learn a mask only over real graph edges.  As in tunedGNN, GCNConv adds an
    # unpruned self-loop for every node during normalization; self-loops are
    # therefore outside MoG's requested edge budget.
    indices = to_undirected(remove_self_loops(data.edge_index)[0])

    if os.environ.get('BASELINE_EVAL_GRAPH', 'sparse').lower() == 'original':
        print(
            '[EvaluationGraph] topology=original-full '
            'training_topology=learned-sparse-mask',
            flush=True,
        )

    print(indices.shape)


    values = torch.ones((indices.size(1),),device=device) # ones
    shape = (data.num_nodes,data.num_nodes)
    #topo_val =data.topo_val
    topo_val = 1

    logger = Logger(args['runs'], args)

    EpochTimes = []

    for run in range(args['runs']):
        select_pyg_split(data, run)
        split_idx = {
            'train': data.train_mask,
            'valid': data.val_mask,
            'test': data.test_mask,
        }
        train_idx = split_idx['train'].to(device)
        # define the model
        model = MoG(data.num_features, num_classes, args, device)
        model = model.to(device)
        
        
        
        optimizer = torch.optim.Adam(
            [
                {
                    'params': model.gnn.parameters(),
                    'lr': args['lr'],
                    'weight_decay': args['weight_decay'],
                },
                {
                    # tunedGNN has no sparsification learner.  Keep its exact
                    # optimizer contract on the GNN while avoiding accidental
                    # regularization of MoG's gating and edge-score tensors.
                    'params': model.learner.parameters(),
                    'lr': args['lr'],
                    'weight_decay': 0.0,
                },
            ]
        )
        print(
            '[OptimizerContract] '
            f'backbone_lr={args["lr"]} '
            f'backbone_weight_decay={args["weight_decay"]} '
            f'learner_lr={args["lr"]} learner_weight_decay=0.0'
        )
        best_val_acc, best_test_acc, best_sparsity = 0, 0, 1
        budget = RunTimeBudget().start()
        if args['use_topo']:
            model.learner.topo_val = topo_val
        else:
            model.learner.topo_val = None
        for epoch in range(1, 1 + args['epochs']):
            # calculate the temperature of sparsity learner
            if (epoch-1) % args["temp_N"] == 0:
                decay_temp = np.exp(-1*args["temp_r"]*epoch)
                temp = max(0.05, decay_temp)
            
            start = time.time()

            # train and test
            loss,mask = train(model,features,indices,labels,values,shape,train_idx,temp,optimizer)

            EpochTimes.append(time.time()-start)

            result = test(model,features,indices,labels,values,shape,split_idx,temp,
                          metric=args['metric'])
            train_acc, valid_acc, test_acc, sparsity, train_f1, test_f1 = result
            
            # log and print
            logger.add_result(run, result)
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_test_acc = test_acc
                best_sparsity = sparsity
                best_mask = mask
                
            
            memory_usage = get_gpu_memory_usage()
            if memory_usage:
                print(memory_usage[0],"MB")


            if epoch % args['log_steps'] == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}%, '
                      f'Test: {100 * test_acc:.2f}%, '
                      f'Test F1 Macro: {100 * test_f1:.2f}%, '
                      f'Best Valid: {100 * best_val_acc:.2f}%, '
                      f'Best Test: {100 * best_test_acc:.2f}%, '
                      f'Best Kept Ratio: {100 * best_sparsity:.2f}%,')
            # logger.add_result() has already recorded this epoch, so the
            # statistics below still select the best epoch actually reached.
            if budget.exhausted(run + 1, epoch, args['epochs']):
                break
        #save_edge_index(indices,best_mask,best_sparsity)
        logger.print_statistics(run)
    logger.print_statistics()

    print("Mean epoch time: ",np.mean(EpochTimes))

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    main()
