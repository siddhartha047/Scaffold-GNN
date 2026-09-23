import argparse
import utils
import os, sys
import logging
import glob
from pathlib import Path

SUPPORT_GRAPH_ROOT = Path(os.environ.get("SUPPORT_GRAPH_ROOT", Path(__file__).resolve().parents[3])).resolve()
if str(SUPPORT_GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPPORT_GRAPH_ROOT))
from scaffold_gnn.utils.defaults import DEFAULT_DATA_DIR


def parser_loader():
    parser = argparse.ArgumentParser(description='AdaGLT')
    parser.add_argument('--total_epoch', type=int, default=400)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument("--retain_epoch", type=int, default=300)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input_dropout', type=float, default=0.0)
    parser.add_argument('--hidden_channels', type=int, default=512)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--metric', choices=('acc', 'rocauc'), default='acc')
    parser.add_argument('--pre_linear', type=int, choices=(0, 1), default=0)
    parser.add_argument(
        '--jumping_knowledge',
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument("--device", type=str, default="cuda:6")

    parser.add_argument("--spar_wei", default=False, action='store_true')
    parser.add_argument("--spar_adj", default=False, action='store_true')
    parser.add_argument('--model_save_path', type=str, default='model_ckpt',)
    parser.add_argument('--save', type=str, default='CKPTs',
                        help='experiment name')
    parser.add_argument("--target_adj_spar", type=float, default=27.0) # 21-22
    parser.add_argument("--target_wei_spar", type=int, default=93)
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--use_bn", action="store_true", default=False)
    parser.add_argument("--use_res", action="store_true", default=False)
    parser.add_argument("--use_bn_value", type=int, choices=(0, 1), default=None)
    parser.add_argument("--use_res_value", type=int, choices=(0, 1), default=None)
    parser.add_argument("--use_ln", type=int, choices=(0, 1), default=0)
    parser.add_argument("--e1", type=float, default=5e-5)
    parser.add_argument("--e2", type=float, default=1e-3)
    parser.add_argument("--coef", type=float, default=0.1)
    parser.add_argument("--task_type", type=str, default="semi")
    parser.add_argument('--seed', type=int, default=None)

    args = vars(parser.parse_args())
    if args['spar_wei']:
        parser.error(
            "The shared AdaGLT benchmark is edge-only; model-weight "
            "sparsification is disabled."
        )
    os.environ["BASELINE_DATA_ROOT"] = args["data_root"]
    seed_dict = {'cora': 1899, 'citeseer': 17889, 'pubmed': 3333}
    # seed_dict = {'cora': 23977/23388, 'citeseer': 27943/27883, 'pubmed': 3333}
    dataset_key = args['dataset'].lower()
    if args['seed'] is None:
        args['seed'] = seed_dict.get(dataset_key, 1899)

    if args['num_layers'] is None:
        args['num_layers'] = 2
    if args['use_bn_value'] is not None:
        args['use_bn'] = bool(args['use_bn_value'])
    if args['use_res_value'] is not None:
        args['use_res'] = bool(args['use_res_value'])

    base_dim = utils.infer_embedding_dim(args['data_root'], args['dataset'])
    args['out_channels'] = base_dim[-1]
    args['embedding_dim'] = (
        [base_dim[0]]
        + [args['hidden_channels']] * args['num_layers']
    )

    args["model_save_path"] = os.path.join(
        args["save"], args["model_save_path"])
    utils.create_exp_dir(args["save"], scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    return args
