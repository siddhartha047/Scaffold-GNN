import torch
import torch.nn as nn
from abc import ABC
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pdb
import torch.nn.init as init
import math
# net_gcn.ginlayers[0].apply_func.mlp.linear
# 
# def soft_threshold(w, th):
# 	'''
# 	pytorch soft-sign function
# 	'''
# 	with torch.no_grad():
# 		temp = torch.abs(w) - th
# 		# print('th:', th)
# 		# print('temp:', temp.size())
# 		return torch.sign(w) * nn.functional.relu(temp)
def prune_adj(oriadj, non_zero_idx, percent):
    
    original_prune_num = int((non_zero_idx / 2) * (percent/100))
    adj = np.copy(oriadj)
    #print("percent:", percent)
    low_adj= np.tril(adj, -1)
    non_zero_low_adj = low_adj[low_adj != 0]
    low_pcen = np.percentile(abs(non_zero_low_adj), percent)
    #print("percentile " + str(low_pcen))
    under_threshold = abs(low_adj) < low_pcen
    before = len(non_zero_low_adj)
    low_adj[under_threshold] = 0
    non_zero_low_adj = low_adj[low_adj != 0]
    after = len(non_zero_low_adj)
    rest_pruned = original_prune_num - (before - after)
    if rest_pruned > 0:
        mask_low_adj = (low_adj != 0)
        low_adj[low_adj == 0] = 2000000
        flat_indices = np.argpartition(low_adj.ravel(), rest_pruned - 1)[:rest_pruned]
        row_indices, col_indices = np.unravel_index(flat_indices, low_adj.shape)
        low_adj = np.multiply(low_adj, mask_low_adj)
        low_adj[row_indices, col_indices] = 0
    adj = low_adj + np.transpose(low_adj)
    adj = np.add(adj, np.identity(adj.shape[0]))

    return adj

def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class AddTrainableMask(ABC):

    _tensor_name: str
    
    def __init__(self):
        pass
    
    def __call__(self, module, inputs):

        setattr(module, self._tensor_name, self.apply_mask(module))

    def apply_mask(self, module):

        mask_train = getattr(module, self._tensor_name + "_mask_train")
        mask_fixed = getattr(module, self._tensor_name + "_mask_fixed")
        orig_weight = getattr(module, self._tensor_name + "_orig_weight")
        pruned_weight = mask_train * mask_fixed * orig_weight 
        
        return pruned_weight

    @classmethod
    def apply(cls, module, name, mask_train, mask_fixed, *args, **kwargs):

        method = cls(*args, **kwargs)  
        method._tensor_name = name
        orig = getattr(module, name)

        module.register_parameter(name + "_mask_train", mask_train.to(dtype=orig.dtype))
        module.register_parameter(name + "_mask_fixed", mask_fixed.to(dtype=orig.dtype))
        module.register_parameter(name + "_orig_weight", orig)
        del module._parameters[name]

        setattr(module, name, method.apply_mask(module))
        module.register_forward_pre_hook(method)

        return method


def add_mask(model, init_mask_dict=None):
    """Attach trainable/fixed weight masks to every GCN layer."""

    for index, layer in enumerate(model.net_layer):
        if init_mask_dict is None:
            mask_train = nn.Parameter(torch.ones_like(layer.weight))
            mask_fixed = nn.Parameter(
                torch.ones_like(layer.weight), requires_grad=False
            )
        else:
            mask_train = nn.Parameter(
                init_mask_dict[f"mask{index + 1}_train"]
            )
            mask_fixed = nn.Parameter(
                init_mask_dict[f"mask{index + 1}_fixed"],
                requires_grad=False,
            )
        AddTrainableMask.apply(
            layer,
            "weight",
            mask_train,
            mask_fixed,
        )
 
        
def generate_mask(model):

    return {
        f"mask{index + 1}": torch.zeros_like(layer.weight)
        for index, layer in enumerate(model.net_layer)
    }


def subgradient_update_mask(model, args):

    if model.adj_mask1_train.grad is not None:
        model.adj_mask1_train.grad.data.add_(
            args['s1'] * torch.sign(model.adj_mask1_train.data)
        )
    for layer in model.net_layer:
        if layer.weight_mask_train.grad is not None:
            layer.weight_mask_train.grad.data.add_(
                args['s2'] * torch.sign(layer.weight_mask_train.data)
            )


def subgradient_update_edge_mask(model, args):
    """Apply Unified-LTH's L1 subgradient only to the graph mask."""

    if model.adj_mask1_train.grad is not None:
        model.adj_mask1_train.grad.data.add_(
            args['s1'] * torch.sign(model.adj_mask1_train.data)
        )


def get_mask_distribution(model, if_numpy=True):

    adj_mask_tensor = model.adj_mask1_train.flatten()
    nonzero = torch.abs(adj_mask_tensor) > 0
    adj_mask_tensor = adj_mask_tensor[nonzero] # 13264 - 2708

    weight_masks = []
    for layer in model.net_layer:
        weight_mask = layer.weight_mask_train.flatten()
        nonzero = torch.abs(weight_mask) > 0
        weight_masks.append(weight_mask[nonzero])
    weight_mask_tensor = torch.cat(weight_masks)
    # np.savez('mask', adj_mask=adj_mask_tensor.detach().cpu().numpy(), weight_mask=weight_mask_tensor.detach().cpu().numpy())
    if if_numpy:
        return adj_mask_tensor.detach().cpu().numpy(), weight_mask_tensor.detach().cpu().numpy()
    else:
        return adj_mask_tensor.detach().cpu(), weight_mask_tensor.detach().cpu()
    

def plot_mask_distribution(model, epoch, acc_test, path):

    print("Plot Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))
    if not os.path.exists(path): os.makedirs(path)
    adj_mask, weight_mask = get_mask_distribution(model)

    plt.figure(figsize=(15, 5))
    plt.subplot(1,2,1)
    plt.hist(adj_mask)
    plt.title("adj mask")
    plt.xlabel('mask value')
    plt.ylabel('times')

    plt.subplot(1,2,2)
    plt.hist(weight_mask)
    plt.title("weight mask")
    plt.xlabel('mask value')
    plt.ylabel('times')
    plt.suptitle("Epoch:[{}] Test Acc[{:.2f}]".format(epoch, acc_test * 100))
    plt.savefig(path + '/mask_epoch{}.png'.format(epoch))


def get_each_mask(mask_weight_tensor, threshold):
    
    ones  = torch.ones_like(mask_weight_tensor)
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, ones, zeros)
    return mask


def _exact_topk_mask(scores, support, keep_count):
    """Return a binary mask with exactly ``keep_count`` supported entries."""

    flat_scores = scores.detach().abs().flatten()
    flat_support = support.detach().bool().flatten()
    active_indices = torch.nonzero(flat_support, as_tuple=False).flatten()
    keep_count = max(0, min(int(keep_count), active_indices.numel()))
    flat_mask = torch.zeros_like(flat_scores)
    if keep_count:
        active_scores = flat_scores[active_indices]
        chosen = torch.topk(
            active_scores,
            k=keep_count,
            largest=True,
            sorted=False,
        ).indices
        flat_mask[active_indices[chosen]] = 1
    return flat_mask.reshape_as(scores)


def _exact_undirected_topk_mask(scores, support, keep_count):
    """Select exact symmetric edge pairs when the support is undirected."""

    support_bool = support.detach().bool()
    if (
        support_bool.ndim != 2
        or support_bool.shape[0] != support_bool.shape[1]
        or not torch.equal(support_bool, support_bool.t())
        or int(keep_count) % 2
    ):
        return _exact_topk_mask(scores, support, keep_count)

    row, col = torch.nonzero(support_bool, as_tuple=True)
    upper_triangle = row < col
    row = row[upper_triangle]
    col = col[upper_triangle]
    pair_keep_count = min(int(keep_count) // 2, row.numel())
    result = torch.zeros_like(scores)
    if pair_keep_count:
        pair_scores = (
            scores.detach().abs()[row, col]
            + scores.detach().abs()[col, row]
        ) / 2.0
        # ``triu_indices`` is in node-id order, so stable sorting also gives
        # deterministic tie breaking without perturbing learned scores.
        chosen = torch.argsort(
            pair_scores,
            descending=True,
            stable=True,
        )[:pair_keep_count]
        selected_row = row[chosen]
        selected_col = col[chosen]
        result[selected_row, selected_col] = 1
        result[selected_col, selected_row] = 1
    return result


def get_final_edge_mask_epoch(model, adj_keep_count):
    """Return the learned exact-cardinality graph ticket only."""

    return {
        "adj_mask": _exact_undirected_topk_mask(
            model.adj_mask1_train,
            model.adj_mask2_fixed.detach(),
            adj_keep_count,
        ).cpu()
    }

def get_each_mask_admm(mask_weight_tensor, threshold):
    
    zeros = torch.zeros_like(mask_weight_tensor) 
    mask = torch.where(mask_weight_tensor.abs() > threshold, mask_weight_tensor, zeros)
    return mask

##### pruning remain mask percent #######
def get_final_mask_epoch(
    model,
    adj_percent,
    wei_percent,
    adj_keep_count=None,
    weight_keep_count=None,
):
    """Select exact-cardinality learned adjacency and model-weight masks."""

    adj_support = model.adj_mask2_fixed.detach()
    active_adj = int(torch.count_nonzero(adj_support).item())
    if adj_keep_count is None:
        adj_keep_count = int(active_adj * (1.0 - adj_percent))
    mask_dict = {
        "adj_mask": _exact_topk_mask(
            model.adj_mask1_train,
            adj_support,
            adj_keep_count,
        ).cpu()
    }

    active_weight_count = sum(
        int(torch.count_nonzero(layer.weight_mask_fixed).item())
        for layer in model.net_layer
    )
    if weight_keep_count is None:
        weight_keep_count = int(
            active_weight_count * (1.0 - wei_percent)
        )
    weight_keep_count = max(
        0, min(int(weight_keep_count), active_weight_count)
    )

    supported_scores = []
    layer_active_indices = []
    for layer in model.net_layer:
        support = layer.weight_mask_fixed.detach().bool().flatten()
        indices = torch.nonzero(support, as_tuple=False).flatten()
        layer_active_indices.append(indices)
        supported_scores.append(
            layer.weight_mask_train.detach().abs().flatten()[indices]
        )

    combined_scores = torch.cat(supported_scores)
    combined_keep = torch.zeros_like(combined_scores)
    if weight_keep_count:
        chosen = torch.topk(
            combined_scores,
            k=weight_keep_count,
            largest=True,
            sorted=False,
        ).indices
        combined_keep[chosen] = 1

    offset = 0
    weight_masks = []
    for layer, active_indices in zip(model.net_layer, layer_active_indices):
        layer_mask = torch.zeros_like(
            layer.weight_mask_train.detach().flatten()
        )
        count = active_indices.numel()
        layer_mask[active_indices] = combined_keep[offset : offset + count]
        weight_masks.append(
            layer_mask.reshape_as(layer.weight_mask_train).cpu()
        )
        offset += count

    mask_dict["weight_masks"] = weight_masks
    for index, weight_mask in enumerate(weight_masks):
        mask_dict[f"weight{index + 1}_mask"] = weight_mask
    return mask_dict

######### ADMM get weight mask ##########
def get_final_weight_mask_epoch(model, wei_percent):

    
    weight1 = model.net_layer[0].weight_orig_weight.detach().cpu().flatten()
    weight2 = model.net_layer[1].weight_orig_weight.detach().cpu().flatten()

    weight_mask_tensor = torch.cat([weight1, weight2])

    wei_y, wei_i = torch.sort(weight_mask_tensor.abs())
    wei_total = weight_mask_tensor.shape[0]
    
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    mask_dict = {}
    mask_dict['weight1_mask'] = get_each_mask(model.net_layer[0].state_dict()['weight_orig_weight'], wei_thre)
    mask_dict['weight2_mask'] = get_each_mask(model.net_layer[1].state_dict()['weight_orig_weight'], wei_thre)

    return mask_dict


##### oneshot magnitude pruning #######
def oneshot_weight_magnitude_pruning(model, wei_percent):

    pdb.set_trace()
    model.net_layer[0].weight_mask_train.requires_grad = False
    model.net_layer[1].weight_mask_train.requires_grad = False

    adj_mask, wei_mask = get_mask_distribution(model, if_numpy=False)
    wei_total = wei_mask.shape[0]
    wei_y, wei_i = torch.sort(wei_mask.abs())
    wei_thre_index = int(wei_total * wei_percent)
    wei_thre = wei_y[wei_thre_index]

    weight1_mask = get_each_mask(model.net_layer[0].state_dict()['weight_mask_train'], wei_thre)
    weight2_mask = get_each_mask(model.net_layer[1].state_dict()['weight_mask_train'], wei_thre)

    return mask_dict



##### random pruning #######
def random_pruning(model, adj_percent, wei_percent):

    model.adj_mask1_train.requires_grad = False
    model.net_layer[0].weight_mask_train.requires_grad = False
    model.net_layer[1].weight_mask_train.requires_grad = False

    adj_nonzero = model.adj_mask1_train.nonzero()
    wei1_nonzero = model.net_layer[0].weight_mask_train.nonzero()
    wei2_nonzero = model.net_layer[1].weight_mask_train.nonzero()

    adj_total = adj_nonzero.shape[0]
    wei1_total = wei1_nonzero.shape[0]
    wei2_total = wei2_nonzero.shape[0]

    adj_kept_num = max(1, min(adj_total, int(adj_total * (1.0 - adj_percent))))
    wei1_kept_num = max(1, min(wei1_total, int(wei1_total * (1.0 - wei_percent))))
    wei2_kept_num = max(1, min(wei2_total, int(wei2_total * (1.0 - wei_percent))))
    adj_pruned_num = adj_total - adj_kept_num
    wei1_pruned_num = wei1_total - wei1_kept_num
    wei2_pruned_num = wei2_total - wei2_kept_num

    adj_index = random.sample([i for i in range(adj_total)], adj_pruned_num)
    wei1_index = random.sample([i for i in range(wei1_total)], wei1_pruned_num)
    wei2_index = random.sample([i for i in range(wei2_total)], wei2_pruned_num)

    adj_pruned = adj_nonzero[adj_index].tolist()
    wei1_pruned = wei1_nonzero[wei1_index].tolist()
    wei2_pruned = wei2_nonzero[wei2_index].tolist()

    for i, j in adj_pruned:
        model.adj_mask1_train[i][j] = 0
        model.adj_mask2_fixed[i][j] = 0
    
    for i, j in wei1_pruned:
        model.net_layer[0].weight_mask_train[i][j] = 0
        model.net_layer[0].weight_mask_fixed[i][j] = 0
    
    for i, j in wei2_pruned:
        model.net_layer[1].weight_mask_train[i][j] = 0
        model.net_layer[1].weight_mask_fixed[i][j] = 0
    
    model.adj_mask1_train.requires_grad = True
    model.net_layer[0].weight_mask_train.requires_grad = True
    model.net_layer[1].weight_mask_train.requires_grad = True

    
def print_sparsity(model):

    adj_nonzero = model.adj_nonzero
    adj_mask_nonzero = model.adj_mask2_fixed.sum().item()
    adj_spar = adj_mask_nonzero * 100 / adj_nonzero

    weight_total = sum(
        layer.weight_mask_fixed.numel() for layer in model.net_layer
    )
    weight_nonzero = sum(
        layer.weight_mask_fixed.sum().item() for layer in model.net_layer
    )

    wei_spar = weight_nonzero * 100 / weight_total
    print("-" * 100)
    print("Sparsity: Adj:[{:.2f}%] Wei:[{:.2f}%]"
    .format(adj_spar, wei_spar))
    print("-" * 100)

    return adj_spar, wei_spar


def print_edge_sparsity(model):
    """Report graph retention while explicitly keeping every model weight."""

    adjacency_kept = model.adj_mask2_fixed.sum().item()
    adjacency_percent = adjacency_kept * 100 / model.adj_nonzero
    print("-" * 100)
    print(
        "Sparsity: Adj:[{:.2f}%] Wei:[100.00%]".format(
            adjacency_percent
        )
    )
    print("-" * 100)
    return adjacency_percent

def print_weight_sparsity(model):

    weight_total = sum(
        layer.weight_mask_fixed.numel() for layer in model.net_layer
    )
    weight_nonzero = sum(
        layer.weight_mask_fixed.sum().item() for layer in model.net_layer
    )

    wei_spar = weight_nonzero * 100 / weight_total
    print("-" * 100)
    print("Sparsity: Wei:[{:.2f}%]".format(wei_spar))
    print("-" * 100)

    return wei_spar


def load_only_mask(model, all_ckpt):

    model_state_dict = model.state_dict()
    masks_state_dict = {k : v for k, v in all_ckpt.items() if 'mask' in k}
    model_state_dict.update(masks_state_dict)
    model.load_state_dict(model_state_dict)


def add_trainable_mask_noise(model, c):
    trainable_masks = [model.adj_mask1_train] + [
        layer.weight_mask_train for layer in model.net_layer
    ]
    with torch.no_grad():
        for mask in trainable_masks:
            noise = (2 * torch.rand_like(mask) - 1) * c
            mask.add_(noise * mask)

    
def soft_mask_init(model, init_type, seed):

    setup_seed(seed)
    mask_pairs = [
        (model.adj_mask1_train, model.adj_mask2_fixed)
    ] + [
        (layer.weight_mask_train, layer.weight_mask_fixed)
        for layer in model.net_layer
    ]

    def initialize(train_mask, fixed_mask, initializer):
        with torch.no_grad():
            initializer(train_mask)
            train_mask.mul_(fixed_mask)

    if init_type == 'all_one':
        add_trainable_mask_noise(model, c=1e-5)
    elif init_type == 'kaiming':
        for train_mask, fixed_mask in mask_pairs:
            initialize(
                train_mask,
                fixed_mask,
                lambda tensor: init.kaiming_uniform_(
                    tensor, a=math.sqrt(5)
                ),
            )
    elif init_type == 'normal':
        mean = 1.0
        std = 0.1
        for train_mask, fixed_mask in mask_pairs:
            initialize(
                train_mask,
                fixed_mask,
                lambda tensor: init.normal_(
                    tensor, mean=mean, std=std
                ),
            )
    elif init_type == 'uniform':
        a = 0.8
        b = 1.2
        for train_mask, fixed_mask in mask_pairs:
            initialize(
                train_mask,
                fixed_mask,
                lambda tensor: init.uniform_(tensor, a=a, b=b),
            )
    else:
        raise ValueError(
            "init_type must be one of all_one, kaiming, normal, uniform"
        )


def soft_edge_mask_init(model, init_type, seed):
    """Initialize only the trainable adjacency mask.

    The fixed mask carries the surviving graph support between IMP rounds.
    Model parameters are normal dense tensors and are deliberately untouched.
    """

    del seed
    train_mask = model.adj_mask1_train
    fixed_mask = model.adj_mask2_fixed
    with torch.no_grad():
        if init_type == "all_one":
            train_mask.copy_(fixed_mask)
        elif init_type == "kaiming":
            init.kaiming_uniform_(train_mask, a=math.sqrt(5))
            train_mask.mul_(fixed_mask)
        elif init_type == "normal":
            init.normal_(train_mask, mean=1.0, std=0.1)
            train_mask.mul_(fixed_mask)
        elif init_type == "uniform":
            init.uniform_(train_mask, a=0.8, b=1.2)
            train_mask.mul_(fixed_mask)
        else:
            raise ValueError(f"Unknown edge-mask initialization: {init_type}")
