import os
import sys, getopt
import argparse
import networkx as nx
# from networkx.algorithms.approximation import independent_set
import numpy as np
from scipy.io import savemat
from scipy.spatial import distance_matrix
import dwave_networkx as dnx
from itertools import chain, combinations
from graph_util import *

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default="./data/Random_Graph_Nb", type=str, help="output directory.")
parser.add_argument("--dist", default="uniform", type=str, help="weight distribution: uniform, normal_l1, normal_l2.")
parser.add_argument("--nbs", default="10, 20, 40, 80, 100, 120, 150", type=str, help="list of average numbers of neighbors.")
parser.add_argument("--ps", default="", type=str, help="list of densities.")
parser.add_argument("--sizes", default="200, 400, 600, 800, 1000", type=str, help="list of numbers of vertices.")
parser.add_argument("--n", default=100, type=int, help="number of instances per configuration.")
parser.add_argument("--bf", default=False, type=bool, help="if use brute force search.")
parser.add_argument("--type", default='ER', type=str, help="ER graph: ER; Poisson: PPP.")
args = parser.parse_args()



def generate_single_config(N, p, N_test, dist_dict, dist, datapath):
    for i in range(N_test):
        filename = '{}_n{}_p{}_b{}_{}.mat'.format(args.type, N, p, i, dist_dict[dist])
        filepath = os.path.join(datapath, filename)
        print("Generating {}".format(filename))
        if args.type.lower() == 'er':
            graph = weighted_random_graph(N, p, dist)
        elif args.type.lower() == 'ppp':
            density = N * 0.01
            r = (10 * np.sqrt(p)) / (np.sqrt(3.1415926) - 2 * np.sqrt(p))
            graph = weighted_poisson_graph(100, density, radius=r, dist=dist)
        elif args.type.lower() == 'ba':
            graph = weighted_barabasi_albert_graph(N, p, dist)
        else:
            continue
        mwis2, val2 = mwis_heuristic_2(graph)
        mwis1, val1 = mwis_heuristic_1(graph)
        mwis0, val0 = mwis_heuristic_greedy(graph)
        # if args.bf:
        #     mwis, val = mwis_bruteforce(graph)
        # if not args.bf:
        if val1 > val2:
            mwis = mwis1
            val = val1
        else:
            mwis = mwis2
            val = val2
        adj_0 = nx.adj_matrix(graph)
        wts = np.array([graph.nodes[u]['weight'] for u in graph.nodes])
        mwis_label = np.zeros((len(graph),), dtype=np.float)
        mwis_label[mwis] = 1
        savemat(filepath, {'adj': adj_0.astype(np.float), 'weights': wts, 'N': N, 'p': p, 'mwis_label': mwis_label,
                           'mwis_utility': val, 'greedy_utility': val0})


def main(args):
    dist = args.dist.lower()
    dist_dict = {'uniform': 'uni', 'normal_l1': 'nl1', 'normal_l2': 'nl2'}
    size_list = [int(item) for item in args.sizes.split(',')]
    nb_list = [float(item) for item in args.nbs.split(',')]
    try:
        p_list = [float(item) for item in args.ps.split(',')]
    except:
        p_list = []
    # datapath = './data/Random_Graph_Nb20'
    # datapath = './data/Random_Graph_Nb100'
    datapath = args.datapath
    if not os.path.isdir(datapath):
        os.mkdir(datapath)

    N_test = args.n  # 50 #10
    # N_test = 1000
    correctness = {}
    maxweights = {}
    Nb_Avgs = [100]

    # for N in [403, 1209]:
    #     for p in [0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05]:
    for N in size_list:
        if len(p_list) == 0:
            for Nb in nb_list:
                p = round(Nb/N, 3)
                generate_single_config(N, p, N_test, dist_dict, dist, datapath)
        else:
            for p in p_list:
                generate_single_config(N, p, N_test, dist_dict, dist, datapath)


if __name__ == "__main__":
    main(args)

