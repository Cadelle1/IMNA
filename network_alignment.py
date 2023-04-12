from input.dataset import Dataset
from algorithms import *
from evaluation.metrics import get_statistics
import utils.graph_utils as graph_utils
import random
import numpy as np
import torch
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--source_dataset', default="graph_data/douban/online/graphsage/")
    parser.add_argument('--target_dataset', default="graph_data/douban/offline/graphsage/")
    parser.add_argument('--groundtruth', default="graph_data/douban/dictionaries/groundtruth")
    parser.add_argument('--seed', default=123, type=int)
    subparsers = parser.add_subparsers(dest="algorithm",
                                       help='Choose 1 of the algorithm from: IsoRank, FINAL, UniAlign, NAWAL, DeepLink, REGAL, IONE, PALE')

    # NAME
    parser_NAME = subparsers.add_parser("NAME", help="NAME algorithm")
    parser_NAME.add_argument('--cuda', action="store_false")
    parser_NAME.add_argument('--embedding_dim', default=192, type=int)
    parser_NAME.add_argument('--NAME_epochs', default=20, type=int)
    parser_NAME.add_argument('--lr', default=0.03, type=float)
    parser_NAME.add_argument('--num_GCN_blocks', type=int, default=2)
    parser_NAME.add_argument('--act', type=str, default='tanh')
    parser_NAME.add_argument('--log', action="store_true", help="Just to print loss")
    parser_NAME.add_argument('--invest', action="store_true", help="To do some statistics")
    parser_NAME.add_argument('--input_dim', default=100, help="Just ignore it")
    parser_NAME.add_argument('--train_dict', type=str)
    parser_NAME.add_argument('--alpha0', type=float, default=1)
    parser_NAME.add_argument('--alpha1', type=float, default=1)
    parser_NAME.add_argument('--alpha2', type=float, default=1)
    parser_NAME.add_argument('--source_embedding')
    parser_NAME.add_argument('--target_embedding')

    # refinement
    parser_NAME.add_argument('--refinement_epochs', default=10, type=int)
    parser_NAME.add_argument('--refine', action="store_true", help="wheather to use refinement step")
    parser_NAME.add_argument('--threshold_refine', type=float, default=0.95,
                             help="The threshold value to get stable candidates")
    # augmentation, let noise_level = 0 if dont want to use it
    parser_NAME.add_argument('--noise_level', default=0, type=float, help="noise to add to augment graph")
    parser_NAME.add_argument('--coe_consistency', default=0.2, type=float, help="consistency weight")
    parser_NAME.add_argument('--threshold', default=0.02, type=float,
                             help="Threshold of for sharpenning")
    parser_NAME.add_argument('--embedding_name', default='')
    parser_NAME.add_argument('--pale_emb_lr', type=float, default=0.01)
    parser_NAME.add_argument('--pale_map_lr', type=float, default=0.01)
    parser_NAME.add_argument('--pale_emb_epochs', type=int, default=500)
    parser_NAME.add_argument('--pale_map_epochs', type=int, default=500)
    parser_NAME.add_argument('--pale_emb_batchsize', type=int, default=512)
    parser_NAME.add_argument('--num_parts', type=int, default=8)
    parser_NAME.add_argument('--mincut_lr', type=float, default=0.001)
    parser_NAME.add_argument('--temp', type=float, default=1)
    parser_NAME.add_argument('--mincut_epochs', type=int, default=2000)
    parser_NAME.add_argument('--hard', action='store_true')
    parser_NAME.add_argument('--beta', type=float, default=1)
    parser_NAME.add_argument('--balance_node', action='store_true')
    parser_NAME.add_argument('--lam', type=float, default=0.99999)
    parser_NAME.add_argument('--w2v_lam', type=float, default=0)
    parser_NAME.add_argument('--new', action='store_true')
    parser_NAME.add_argument('--clip', type=float, default=2.0)
    parser_NAME.add_argument('--anneal', action='store_true')
    parser_NAME.add_argument('--min_temp', type=float, default=0.1)
    parser_NAME.add_argument('--debug', action='store_true')
    parser_NAME.add_argument('--file', type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    start_time = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    source_dataset = Dataset(args.source_dataset)
    target_dataset = Dataset(args.target_dataset)
    groundtruth = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')

    algorithm = args.algorithm

    model = NAME(source_dataset, target_dataset, args)

    S = model.align()
    end_time = time.time()
    print("run_time:",end_time-start_time)

    acc, MAP, Hit, AUC, top5, top10, top20, top30, top50, top100 = get_statistics(S, groundtruth, get_all_metric=True)
    print("Top_1: {:.4f}".format(acc))
    print("Top_5: {:.4f}".format(top5))
    print("Top_10: {:.4f}".format(top10))
    print("Top_20: {:.4f}".format(top20))
    print("Top_30: {:.4f}".format(top30))
    print("Hit: {:.4f}".format(Hit))
    print("MAP: {:.4f}".format(MAP))
    print("AUC: {:.4f}".format(AUC))
    print("-" * 100)

    result = acc, top5, top10, top20, top30
