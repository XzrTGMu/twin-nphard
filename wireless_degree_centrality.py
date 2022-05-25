#!/usr/bin/ python3
# -*- coding: utf-8 -*-
# python3
# Make this standard template for testing and training
import networkx as nx
# from networkx.algorithms.approximation import independent_set
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import time
from collections import deque
from copy import deepcopy
from scipy.io import savemat
from scipy.spatial import distance_matrix
import dwave_networkx as dnx
import sys
import os
from copy import copy, deepcopy
from itertools import chain, combinations
from heuristics import greedy_search, dist_greedy_search, local_greedy_search, mlp_gurobi
# visualization
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from graph_util import *
# from test_utils import *

from runtime_config import flags
flags.DEFINE_string('output', 'wireless', 'output folder')
flags.DEFINE_string('test_datapath', './data/ER_Graph_Uniform_NP20_test', 'test dataset')
flags.DEFINE_string('wt_sel', 'qr', 'qr: queue length * rate, q/r: q/r, q: queue length only, otherwise: random')
flags.DEFINE_float('load_min', 0.01, 'traffic load min')
flags.DEFINE_float('load_max', 0.15, 'traffic load max')
flags.DEFINE_float('load_step', 0.01, 'traffic load step')
flags.DEFINE_integer('instances', 10, 'number of layers.')
flags.DEFINE_integer('num_channels', 1, 'number of channels')
flags.DEFINE_integer('opt', 0, 'test algorithm')
flags.DEFINE_string('graph', 'poisson', 'type of graphs')


n_instances = flags.FLAGS.instances


gtype = flags.FLAGS.graph
train = False
n_networks = 500
# n_instances = 10
timeslots = 64
lp = 5

sim_area = 250
sim_node = 100
sim_rc = 1
sim_ri = 4
n_ch = 1
p_overlap = 0.8
# link rate high and low bound (number of packets per time slot)
sim_rate_hi = 100
sim_rate_lo = 0
# Testing load range (upper limit = 1/(average degree of conflict graphs))
# 10.78 for 10 graphs, 10.56 for 20 graphs
load_min = flags.FLAGS.load_min
load_max = flags.FLAGS.load_max
load_step = flags.FLAGS.load_step
wt_sel = flags.FLAGS.wt_sel


output_dir = flags.FLAGS.output
output_csv = os.path.join(output_dir,
                          'degree_centrality_{}.csv'
                          .format(gtype)
                          )

res_list = []
res_df = pd.DataFrame(columns=['graph',
                               'seed',
                               'name',
                               'degree_centrality',
                               'tmh', 'skewness', 'kurtosis', 'par'])
# if os.path.isfile(output_csv):
#     res_df = pd.read_csv(output_csv, index_col=0)


if train:
    datapath = flags.FLAGS.datapath
    epochs = flags.FLAGS.epochs
else:
    datapath = flags.FLAGS.test_datapath
    epochs = 1

val_mat_names = sorted(os.listdir(datapath))

cnt = 0

np.random.seed(1)
if train:
    loss = 1.0
else:
    loss = np.nan

wts_sample_file = os.path.join(output_dir, 'samples.txt')

load_array = np.round(np.arange(load_min, load_max+load_step, load_step), 2)
# load = load_array[np.random.randint(0, len(load_array) - 1)]

degree_centrality = []
for i in range(500):
    if gtype == 'poisson':
        if i >= len(val_mat_names):
            break
        idx = i
        mat_contents = sio.loadmat(os.path.join(datapath, val_mat_names[idx]))
        gdict = mat_contents['gdict'][0, 0]
        seed = mat_contents['random_seed'][0, 0]
        graph_c, graph_i = poisson_graphs_from_dict(gdict)
        adj_gK = nx.adjacency_matrix(graph_i)
        flows = [e for e in graph_c.edges]
        # flows_r = [(e[1], e[0]) for e in graph_c.edges]
        # flows = flows + flows_r
        nflows = len(flows)
    elif gtype == 'star30':
        graph_i = nx.star_graph(30)
        adj_gK = nx.adjacency_matrix(graph_i)
        nflows = adj_gK.shape[0]
        seed = i
    elif gtype == 'star20':
        graph_i = nx.star_graph(20)
        adj_gK = nx.adjacency_matrix(graph_i)
        nflows = adj_gK.shape[0]
        seed = i
    elif gtype == 'star10':
        graph_i = nx.star_graph(10)
        adj_gK = nx.adjacency_matrix(graph_i)
        nflows = adj_gK.shape[0]
        seed = i
    elif gtype == 'ba1':
        graph_i = nx.barabasi_albert_graph(70, 1)
        adj_gK = nx.adjacency_matrix(graph_i)
        nflows = adj_gK.shape[0]
        seed = i
    elif gtype == 'ba2':
        graph_i = nx.barabasi_albert_graph(70, 2)
        adj_gK = nx.adjacency_matrix(graph_i)
        nflows = adj_gK.shape[0]
        seed = i
    elif gtype == 'er':
        graph_i = nx.erdos_renyi_graph(50, 0.1)
        adj_gK = nx.adjacency_matrix(graph_i)
        nflows = adj_gK.shape[0]
        seed = i
    elif gtype == 'tree':
        try:
            graph_i = nx.random_powerlaw_tree(50, gamma=3.0, seed=i, tries=2000)
        except:
            graph_i = nx.random_powerlaw_tree(50, gamma=3.0, tries=1000)
        adj_gK = nx.adjacency_matrix(graph_i)
        nflows = adj_gK.shape[0]
        seed = i
    else:
        if i >= len(val_mat_names):
            break
        idx = i
        mat_contents = sio.loadmat(os.path.join(datapath, val_mat_names[idx]))
        adj_gK = mat_contents['adj']
        wts = mat_contents['weights'].transpose()
        nflows = adj_gK.shape[0]
        seed = i
        graph_i = nx.from_scipy_sparse_matrix(adj_gK)
    netcfg = "Config: s {}, n {}, f {}, t {}".format(seed, sim_node, nflows, timeslots)

    # degree_cent = nx.degree_centrality(graph_i)

    degree_cent, degs = degree_centralization(graph_i)
    tmh = np.sum(np.power(degs,2))/np.sum(degs)
    kurtosis = stats.kurtosis(degs)
    skewness = stats.skew(degs)
    par = np.amax(degs)/np.mean(degs)

    degree_centrality.append(degree_cent)

    res_df = res_df.append({'graph': i,
                            'seed': seed,
                            'name': gtype,
                            'degree_centrality': degree_cent,
                            'tmh': tmh,
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'par': par
                            }, ignore_index=True)

    lg = nx.line_graph(graph_i)
    degree_cent, degs = degree_centralization(lg)
    tmh = np.sum(np.power(degs,2))/np.sum(degs)
    kurtosis = stats.kurtosis(degs)
    skewness = stats.skew(degs)
    par = np.amax(degs)/np.mean(degs)

    degree_centrality.append(degree_cent)

    res_df = res_df.append({'graph': i,
                            'seed': seed,
                            'name': gtype+'-line',
                            'degree_centrality': degree_cent,
                            'tmh': tmh,
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'par': par
                            }, ignore_index=True)


print("{}: {}, ".format(500, gtype),
      "deg cent avg: {:.3f}, ".format(np.nanmean(degree_centrality)),
      "TMH: {:.3f}".format(np.nanmean(res_df['tmh'])),
      "Skewness: {:.3f}".format(np.nanmean(res_df['skewness'])),
      "Kurtosis: {:.3f}".format(np.nanmean(res_df['kurtosis'])),
      "Par: {:.3f}".format(np.nanmean(res_df['par']))
      )

res_df.to_csv(output_csv, index=False)
print("Done!")

