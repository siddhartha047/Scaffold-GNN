import torch
import torch.nn.functional as F
from .dataset import canonicalize_dataset_name

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        out = model(
            dataset.graph['node_feat'],
            dataset.graph['edge_index'],
            dataset.graph.get('edge_weight'),
        )
    train_acc = eval_func(dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(dataset.label[split_idx['test']], out[split_idx['test']])
    if canonicalize_dataset_name(args.dataset) in ('questions', 'ogbn-proteins'):
        if dataset.label.ndim == 1 or dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label.reshape(-1), dataset.label.max() + 1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[split_idx['valid']].to(torch.float))
    else:
        log_out = F.log_softmax(out, dim=1)
        labels = dataset.label.squeeze()
        valid_loss = criterion(log_out[split_idx['valid']], labels[split_idx['valid']])
    return (train_acc, valid_acc, test_acc, valid_loss, out)

@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None):
    model_cpu = model.to(torch.device('cpu'))
    label_cpu = dataset.label.to(torch.device('cpu'))
    edge_index = dataset.graph['edge_index'].to(torch.device('cpu'))
    edge_weight = dataset.graph.get('edge_weight')
    if edge_weight is not None:
        edge_weight = edge_weight.to(torch.device('cpu'))
    x = dataset.graph['node_feat'].to(torch.device('cpu'))
    if result is not None:
        out = result
    else:
        model.eval()
        out = model_cpu(x, edge_index, edge_weight)
    train_acc = eval_func(label_cpu[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(label_cpu[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(label_cpu[split_idx['test']], out[split_idx['test']])
    if canonicalize_dataset_name(args.dataset) in ('questions', 'ogbn-proteins'):
        if label_cpu.ndim == 1 or label_cpu.shape[1] == 1:
            true_label = F.one_hot(label_cpu.reshape(-1), label_cpu.max() + 1)
        else:
            true_label = label_cpu
        valid_loss = criterion(out[split_idx['valid']], true_label[split_idx['valid']].float())
    else:
        log_out = F.log_softmax(out, dim=1)
        labels = label_cpu.squeeze()
        valid_loss = criterion(log_out[split_idx['valid']], labels[split_idx['valid']])
    model.to(device)
    return (train_acc, valid_acc, test_acc, valid_loss, out)
