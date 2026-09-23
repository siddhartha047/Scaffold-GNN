import torch
import torch.nn.functional as F
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

def rand_train_test_idx(label, train_prop=0.5, valid_prop=0.25, ignore_negative=True, generator=None):
    label = label.detach().cpu()
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = torch.arange(label.shape[0])
    n = labeled_nodes.shape[0]
    perm = torch.randperm(n, generator=generator)
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    train_idx = labeled_nodes[perm[:train_num]]
    valid_idx = labeled_nodes[perm[train_num:train_num + valid_num]]
    test_idx = labeled_nodes[perm[train_num + valid_num:]]
    return (train_idx, valid_idx, test_idx)


def balanced_train_valid_test_idx(
    label,
    train_nodes_per_class=2,
    valid_num=8,
    ignore_negative=True,
    generator=None,
):
    """Use a fixed training count per class and all remaining nodes for eval."""
    label = label.detach().cpu().view(-1)
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = torch.arange(label.shape[0])

    classes = torch.unique(label[labeled_nodes], sorted=True)
    train_parts = []
    remaining_by_class = []
    for class_id in classes:
        nodes = labeled_nodes[label[labeled_nodes] == class_id]
        if nodes.numel() < train_nodes_per_class:
            raise RuntimeError(
                f"Class {int(class_id)} has {nodes.numel()} nodes, fewer than "
                f"train_nodes_per_class={train_nodes_per_class}."
            )
        nodes = nodes[torch.randperm(nodes.numel(), generator=generator)]
        train_parts.append(nodes[:train_nodes_per_class])
        remaining_by_class.append(nodes[train_nodes_per_class:])

    remaining_counts = torch.tensor(
        [nodes.numel() for nodes in remaining_by_class],
        dtype=torch.long,
    )
    remaining_total = int(remaining_counts.sum())
    valid_num = int(valid_num)
    if valid_num < 0 or valid_num > remaining_total:
        raise ValueError(
            f"valid_num must be between 0 and {remaining_total}, got {valid_num}."
        )

    ideal = remaining_counts.to(torch.float64) * (
        float(valid_num) / max(1, remaining_total)
    )
    valid_allocation = torch.floor(ideal).to(torch.long)
    preserve_test_capacity = torch.clamp(remaining_counts - 1, min=0)
    capacity = (
        preserve_test_capacity
        if valid_num <= int(preserve_test_capacity.sum())
        else remaining_counts
    )
    valid_allocation = torch.minimum(valid_allocation, capacity)

    while int(valid_allocation.sum()) < valid_num:
        best = None
        for class_pos in range(len(remaining_by_class)):
            if int(valid_allocation[class_pos]) >= int(capacity[class_pos]):
                continue
            score = float(ideal[class_pos] - valid_allocation[class_pos])
            candidate = (score, -class_pos, class_pos)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            raise RuntimeError("Unable to allocate the requested validation nodes.")
        valid_allocation[best[-1]] += 1

    valid_parts = []
    test_parts = []
    for class_pos, nodes in enumerate(remaining_by_class):
        count = int(valid_allocation[class_pos])
        valid_parts.append(nodes[:count])
        test_parts.append(nodes[count:])

    def shuffled(parts):
        indices = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
        if indices.numel() > 1:
            indices = indices[torch.randperm(indices.numel(), generator=generator)]
        return indices

    return shuffled(train_parts), shuffled(valid_parts), shuffled(test_parts)


def class_rand_splits(label, label_num_per_class, valid_num=500, test_num=1000):
    label = label.detach().cpu().view(-1)
    train_idx, non_train_idx = ([], [])
    idx = torch.arange(label.shape[0])
    class_list = torch.unique(label[label != -1])
    for c in class_list:
        idx_i = idx[label == c]
        n_i = idx_i.shape[0]
        if n_i < label_num_per_class:
            raise RuntimeError('Insufficient samples in a class.')
        perm = idx_i[torch.randperm(n_i)]
        train_idx += perm[:label_num_per_class].tolist()
        non_train_idx += perm[label_num_per_class:].tolist()
    train_idx = torch.as_tensor(train_idx)
    non_train_idx = torch.as_tensor(non_train_idx)
    perm2 = torch.randperm(non_train_idx.shape[0])
    non_train_idx = non_train_idx[perm2]
    valid_idx = non_train_idx[:valid_num]
    test_idx = non_train_idx[valid_num:valid_num + test_num]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}

def _format_dataset_name(name):
    return ''.join([p.capitalize() for p in name.split('-')])

def load_fixed_splits(data_dir, dataset, name):
    splits_lst = []
    if name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        fixed_name = _format_dataset_name(name)
        torch_dataset = HeterophilousGraphDataset(name=fixed_name, root=data_dir)
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits_lst.append({'train': torch.where(data.train_mask[:, i])[0], 'valid': torch.where(data.val_mask[:, i])[0], 'test': torch.where(data.test_mask[:, i])[0]})
    elif name in ['wikics']:
        torch_dataset = WikiCS(root=f'{data_dir}/wikics/')
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits_lst.append({'train': torch.where(data.train_mask[:, i])[0], 'valid': torch.where(data.val_mask[:, i])[0], 'test': torch.where(data.test_mask)[0]})
    elif name in ['amazon-computer', 'amazon-photo', 'coauthor-cs', 'coauthor-physics']:
        idx = np.load(f'{data_dir}/{name}_split.npz')
        splits_lst.append({'train': torch.from_numpy(idx['train']).long(), 'valid': torch.from_numpy(idx['valid']).long(), 'test': torch.from_numpy(idx['test']).long()})
    elif name in ['pokec']:
        split = np.load(f'{data_dir}/{name}/{name}-splits.npy', allow_pickle=True)
        for s in split:
            splits_lst.append({'train': torch.from_numpy(np.asarray(s['train'])).long(), 'valid': torch.from_numpy(np.asarray(s['valid'])).long(), 'test': torch.from_numpy(np.asarray(s['test'])).long()})
    elif name in ['chameleon', 'squirrel']:
        file_path = f'{data_dir}/geom-gcn/{name}/{name}_filtered.npz'
        data = np.load(file_path)
        train_masks = data['train_masks']
        val_masks = data['val_masks']
        test_masks = data['test_masks']
        N = train_masks.shape[1]
        node_idx = np.arange(N)
        for i in range(train_masks.shape[0]):
            splits_lst.append({'train': torch.as_tensor(node_idx[train_masks[i]]), 'valid': torch.as_tensor(node_idx[val_masks[i]]), 'test': torch.as_tensor(node_idx[test_masks[i]])})
    else:
        raise NotImplementedError
    return splits_lst

def eval_f1(y_true, y_pred):
    yt = y_true.detach().cpu().numpy().reshape(-1)
    yp = y_pred.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
    return f1_score(yt, yp, average='micro')

def eval_acc(y_true, y_pred):
    yt = y_true.detach().cpu().numpy().reshape(-1)
    yp = y_pred.argmax(dim=-1).detach().cpu().numpy().reshape(-1)
    return (yt == yp).mean()


def label_count_distribution(labels, num_classes=None):
    """Return node counts per class, or positive-label counts for multi-label data."""
    labels = torch.as_tensor(labels).detach().cpu()
    if labels.ndim > 1 and labels.shape[-1] > 1:
        valid = torch.isfinite(labels) & (labels >= 0)
        positive_counts = ((labels > 0) & valid).sum(dim=0)
        return {
            int(class_id): int(positive_counts[class_id].item())
            for class_id in range(int(labels.shape[-1]))
        }

    labels = labels.reshape(-1).long()
    labels = labels[labels >= 0]
    observed_num_classes = int(labels.max().item()) + 1 if labels.numel() else 0
    resolved_num_classes = (
        max(0, int(num_classes))
        if num_classes is not None
        else observed_num_classes
    )
    counts = torch.bincount(labels, minlength=resolved_num_classes)
    return {
        int(class_id): int(counts[class_id].item())
        for class_id in range(resolved_num_classes)
    }


def eval_acc_per_class(y_true, y_pred, num_classes=None):
    """Return single-label accuracy for each class as fractions in [0, 1]."""
    y_true = torch.as_tensor(y_true).detach().cpu()
    y_pred = torch.as_tensor(y_pred).detach().cpu()

    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        return None

    y_true = y_true.reshape(-1).long()
    if y_pred.ndim > 1:
        if y_pred.shape[-1] == 1:
            predicted_class = (y_pred.reshape(-1) > 0).long()
            inferred_num_classes = 2
        else:
            predicted_class = y_pred.argmax(dim=-1).reshape(-1).long()
            inferred_num_classes = int(y_pred.shape[-1])
    else:
        predicted_class = y_pred.reshape(-1).long()
        inferred_num_classes = 0

    if y_true.numel() != predicted_class.numel():
        raise ValueError(
            'Per-class accuracy requires the same number of labels and predictions: '
            f'got {y_true.numel()} labels and {predicted_class.numel()} predictions.'
        )

    labeled = y_true >= 0
    y_true = y_true[labeled]
    predicted_class = predicted_class[labeled]
    observed_num_classes = int(y_true.max().item()) + 1 if y_true.numel() else 0
    resolved_num_classes = (
        max(0, int(num_classes))
        if num_classes is not None
        else max(inferred_num_classes, observed_num_classes)
    )

    accuracy_by_class = {}
    for class_id in range(resolved_num_classes):
        class_mask = y_true == class_id
        accuracy_by_class[class_id] = (
            float((predicted_class[class_mask] == class_id).float().mean().item())
            if class_mask.any()
            else None
        )
    return accuracy_by_class


def average_acc_per_class(per_run_accuracies):
    """Average per-class accuracy dictionaries, ignoring absent classes."""
    class_ids = sorted({
        int(class_id)
        for run_accuracies in per_run_accuracies
        if run_accuracies is not None
        for class_id in run_accuracies
    })
    averaged = {}
    for class_id in class_ids:
        values = [
            float(run_accuracies[class_id])
            for run_accuracies in per_run_accuracies
            if run_accuracies is not None
            and run_accuracies.get(class_id) is not None
        ]
        averaged[class_id] = float(np.mean(values)) if values else None
    return averaged


def eval_rocauc(y_true, y_pred):
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        yt = y_true.detach().cpu().numpy()
        yp = y_pred.detach().cpu().numpy()
        scores = []
        for k in range(yt.shape[1]):
            is_labeled = np.isfinite(yt[:, k]) & (yt[:, k] >= 0)
            if is_labeled.sum() == 0:
                continue
            yk = yt[is_labeled, k]
            pk = yp[is_labeled, k]
            if np.unique(yk).size < 2:
                continue
            scores.append(roc_auc_score(yk, pk))
        if len(scores) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')
        return float(np.mean(scores))

    yt = y_true.detach().cpu().numpy().reshape(-1)
    classes = np.unique(yt)
    if len(classes) == 2:
        prob = F.softmax(y_pred, dim=-1)[:, 1].detach().cpu().numpy()
        return roc_auc_score(yt, prob)
    prob = F.softmax(y_pred, dim=-1).detach().cpu().numpy()
    yt_bin = label_binarize(yt, classes=classes)
    scores = []
    for k in range(prob.shape[1]):
        if yt_bin[:, k].sum() > 0 and (1 - yt_bin[:, k]).sum() > 0:
            scores.append(roc_auc_score(yt_bin[:, k], prob[:, k]))
    if len(scores) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')
    return np.mean(scores)

def eval_f1_macro(y_true, y_pred):
    y_true_cpu = y_true.detach().cpu()
    y_pred_cpu = y_pred.detach().cpu()
    if y_true_cpu.ndim > 1 and y_true_cpu.shape[-1] > 1:
        # Multi-label tasks (notably OGBN-Proteins) emit independent logits.
        # Match the conventional 0.5 sigmoid threshold by thresholding logits
        # at zero, then macro-average positive-class F1 across valid tasks.
        true_array = y_true_cpu.numpy()
        pred_array = (y_pred_cpu >= 0).numpy().astype(np.int64)
        task_scores = []
        for task in range(true_array.shape[1]):
            valid = np.isfinite(true_array[:, task]) & (true_array[:, task] >= 0)
            if not np.any(valid):
                continue
            task_scores.append(f1_score(
                true_array[valid, task].astype(np.int64),
                pred_array[valid, task],
                average='binary',
                zero_division=0,
            ))
        if not task_scores:
            raise RuntimeError('No labeled tasks available for macro-F1')
        return float(np.mean(task_scores))

    true_array = y_true_cpu.numpy().reshape(-1)
    pred_array = y_pred_cpu.argmax(dim=-1).numpy().reshape(-1)
    return f1_score(true_array, pred_array, average='macro', zero_division=0)
dataset_drive_url = {'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia', 'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y', 'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ'}
