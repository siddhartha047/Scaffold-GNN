import argparse
import os
import sys
from pathlib import Path

SUPPORT_GRAPH_ROOT = Path(os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))
from scaffold_gnn.utils.defaults import DEFAULT_DATA_DIR


def parser_loader():
    parser = argparse.ArgumentParser(description='MoG(citation)')
    # experiment settings
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--temp_N',type=int,default=50)
    parser.add_argument('--temp_r',type=float,default=1e-3)
    parser.add_argument('--seed',type=int,default=5)
    parser.add_argument('--dataset', type=str, default='SmallCora')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_DIR)
    
    # args about optim
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    
    # args about gnn
    parser.add_argument('--hidden_channels',type=int,default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)    
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--metric', choices=('acc', 'rocauc'), default='acc')
    parser.add_argument('--pre_linear', type=int, choices=(0, 1), default=0)
    parser.add_argument('--residual', type=int, choices=(0, 1), default=0)
    parser.add_argument('--layer_norm', type=int, choices=(0, 1), default=0)
    parser.add_argument('--batch_norm', type=int, choices=(0, 1), default=0)
    parser.add_argument('--jumping_knowledge', type=int, choices=(0, 1), default=0)
    
    # args about SpLearner expert
    parser.add_argument('--hidden_spl',type=float,default=128)
    parser.add_argument('--num_layers_spl',type=int,default=2)
    
    # args about MoE
    parser.add_argument('--expert_select',type=int,default=3)
    parser.add_argument('--k_list', nargs='+', type=float)
    parser.add_argument('--lam',type=float,default=1e-1)
    #parser.add_argument('--use_topo',default=True,action="store_true")
    parser.add_argument('--use_topo',default=False,action="store_false")
    
    args = vars(parser.parse_args())
    assert len(args['k_list']),"The sparsity of each sparsifier must be specified"
    
    return args
