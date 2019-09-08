import util
import time
import numpy as np
import scipy
from os.path import join
import argparse
import os
from multiprocessing import Pool
from gntk import GNTK

parser = argparse.ArgumentParser(description='GNTK computation')
# several folders, each folder one kernel
parser.add_argument('--dataset', type=str, default="COLLAB",
                        help='name of dataset (default: COLLAB)')
parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of mlp layers')
parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
parser.add_argument('--scale', type=str, default='degree',
						help='scaling methods')
parser.add_argument('--jk', type=int, default=1,
						help='whether to add jk')
parser.add_argument('--out_dir', type=str, default="out",
                    help='output directory')
args = parser.parse_args()

if args.dataset in ['IMDBBINARY', 'COLLAB', 'IMDBMULTI', 'COLLAB']:
    # social network
    degree_as_tag = True
elif args.dataset in ['MUTAG', 'PROTEINS', 'PTC', 'NCI1']:
    # bioinformatics
    degree_as_tag = False
    
graphs, _  = util.load_data(args.dataset, degree_as_tag)
labels = np.array([g.label for g in graphs]).astype(int)

gntk = GNTK(num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers, jk=args.jk, scale=args.scale)
A_list = []
diag_list = []

# procesing the data
for i in range(len(graphs)):
    n = len(graphs[i].neighbors)
    for j in range(n):
        graphs[i].neighbors[j].append(j)
    edges = graphs[i].g.edges
    m = len(edges)

    row = [e[0] for e in edges]
    col = [e[1] for e in edges]

    A_list.append(scipy.sparse.coo_matrix(([1] * len(edges), (row, col)), shape = (n, n), dtype = np.float32))
    A_list[-1] = A_list[-1] + A_list[-1].T + scipy.sparse.identity(n)
    diag = gntk.diag(graphs[i], A_list[i])
    diag_list.append(diag)


def calc(T):
    return gntk.gntk(graphs[T[0]], graphs[T[1]], diag_list[T[0]], diag_list[T[1]], A_list[T[0]], A_list[T[1]])

calc_list = [(i, j) for i in range(len(graphs)) for j in range(i, len(graphs))]

pool = Pool(80)
results = pool.map(calc, calc_list)

gram = np.zeros((len(graphs), len(graphs)))
for t, v in zip(calc_list, results):
    gram[t[0], t[1]] = v
    gram[t[1], t[0]] = v
    

np.save(join(args.out_dir, 'gram'), gram)
np.save(join(args.out_dir, 'labels'), labels)
