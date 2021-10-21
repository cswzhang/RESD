# coding=utf-8
import argparse
import time
import warnings

import networkx as nx
import torch
from sklearn.exceptions import UndefinedMetricWarning
from torch.utils.data import DataLoader

from models.model import RESD
from utils import *

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(
        "Deep Recursive Network Embedding with Regular Equivalence")
    parser.add_argument('--dataset', type=str, default="barbell",
                        help='Directory to load data.')
    parser.add_argument('-s', '--struct', type=str, default="-1,128,128",
                        help='the network struct')
    parser.add_argument('-e', '--epoch', type=int, default=50,
                        help='Number of epoch to train. Each epoch processes the training '
                             'data once completely')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Number of training examples processed per step')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('-a', '--alpha', type=float, default=1,
                        help='the rate of vae loss')
    parser.add_argument('-g', '--gamma', type=float, default=1,
                        help='the rate of gan-relation loss')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='ratio to split the '
                                                                       'train data')
    parser.add_argument('--loop', type=int, default=100, help='num of classification')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='L2')
    parser.add_argument('--device', type=int, default=0, help='GPU')
    parser.add_argument('--no-classification', dest='classification', action='store_false',
                        help='classification')
    return parser.parse_args()


def _main(args):
    graph_path = 'dataset/clf/{}.edge'.format(args.dataset)
    feature_path = 'cache/features/{}_features.csv'.format(args.dataset)
    lbl_path = 'dataset/clf/{}.lbl'.format(args.dataset)
    G = nx.read_edgelist(graph_path, nodetype=int)

    features = pd.read_csv(feature_path).values

    print("Nodes: {}, Edges: {}".format(G.number_of_nodes(), G.number_of_edges()))
    print("Features size: {}".format(features.shape))
    device = torch.device(
        "cuda:{}".format(args.device) if args.device >= 0 else "cpu")
    args.device = device
    args.struct = list(map(lambda x: int(x), args.struct.split(',')))

    model = RESD(args, G, features).to(device, dtype=torch.float32)
    optimizer = torch.optim.Adam([
        {'params': model.vae.parameters(), 'weight_decay': 0},
        {'params': model.mlp.parameters()},
    ], lr=args.learning_rate, weight_decay=args.weight_decay)
    t1 = time.time()
    print("Start time: {}".format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1))))
    total_time = 0
    for epoch in range(args.epoch):
        train_dataloader = DataLoader(list(G.nodes), args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
        total_loss = 0.0
        start = time.time()

        for idx, data in enumerate(train_dataloader):
            nodes = data
            optimizer.zero_grad()
            loss = model(nodes)
            loss.backward()
            optimizer.step()
            total_loss += loss
        end = time.time()
        total_time += end - start
        print('epoch: {}, Total loss: {}, Time: {}'.format(epoch, total_loss.item(), end - start))

        if args.classification:
            embedding = model.get_embedding()
            embedding = embedding.data.cpu().numpy()
            if (epoch + 1) % 10 == 0:
                classification(embedding, lbl_path, split_ratio=args.split_ratio, loop=args.loop)
    t2 = time.time()
    print("Embedding time: {}".format(t2 - t1))
    print(total_time)
    embedding = model.get_embedding()
    embedding = embedding.data.cpu().numpy()
    classification(embedding, lbl_path, split_ratio=args.split_ratio, loop=args.loop)
    dimension = embedding.shape[1]
    columns = ["id"] + ["x_" + str(x) for x in range(embedding.shape[1])]
    ids = np.array(list(range(embedding.shape[0]))).reshape((-1, 1))
    embedding = pd.DataFrame(np.concatenate([ids, embedding], axis=1), columns=columns)
    embedding = embedding.sort_values(by=['id'])
    if args.gamma == 0:
        print('Save best embedding to embed/RES/{}_{}.emb'.format(args.dataset, dimension))
        embedding.to_csv("embed/RES/{}_{}.emb".format(args.dataset, dimension), index=False)
    else:
        print('Save best embedding to embed/RESD/{}_{}.emb'.format(args.dataset, dimension))
        embedding.to_csv("embed/RESD/{}_{}.emb".format(args.dataset, dimension), index=False)


if __name__ == '__main__':
    _main(parse_args())
