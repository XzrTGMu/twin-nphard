# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )
# add the libary path for graph reduction and local search
# sys.path.append( '%s/kernel' % os.path.dirname(os.path.realpath(__file__)) )

import time
import random
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from multiprocessing import Queue
from copy import deepcopy
from scipy.stats.stats import pearsonr, linregress
from scipy.spatial import distance_matrix

import tensorflow as tf
import networkx as nx
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
# Settings (FLAGS)
from runtime_config import flags, FLAGS
from heuristics_mwcds import *

flags.DEFINE_string('gtype', 'ba', 'test graph type: er, grp, ws, ba')
flags.DEFINE_string('test_datapath', './data/ER_Graph_Uniform_NP20_test', 'test dataset')
flags.DEFINE_integer('ntrain', 1, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('nvalid', 100, 'Number of outputs.')

from steiner_gcn_call_twin import DPGAgent, heuristic_func
dqn_agent = DPGAgent(FLAGS, 5000)


# test data path
data_path = FLAGS.datapath
test_datapath = FLAGS.test_datapath

# Some preprocessing
noout = min(FLAGS.diver_num, FLAGS.diver_out) # number of outputs
time_limit = FLAGS.timeout  # time limit for searching
backoff_thresh = 1 - FLAGS.backoff_prob

num_supports = 1 + FLAGS.max_degree
nsr = np.power(10.0, -FLAGS.snr_db/20.0)

from directory import create_result_folder, find_model_folder
model_origin = find_model_folder(FLAGS, 'dpg_policy')
critic_origin = find_model_folder(FLAGS, 'critic')

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


def random_graph(size, k=20, p=0.25, gtype='grp', gseed=None):
    if gtype == 'grp':
        graph = nx.gaussian_random_partition_graph(size, k, min(7, k), p, max(0.1, p/3.0), seed=gseed)
    elif gtype == 'ws':
        graph = nx.connected_watts_strogatz_graph(size, k, p, tries=1000, seed=gseed)
    elif gtype == 'er':
        graph = nx.generators.random_graphs.fast_gnp_random_graph(size, float(k) / float(size))
    elif gtype == 'ba':
        graph = nx.generators.random_graphs.barabasi_albert_graph(size, int(np.round(k * p)))
    else:
        raise ValueError('Unsupported graph type')
    wts = 10.0*np.random.uniform(0.01, 1.01, (size,))
    for u in graph:
        graph.nodes[u]['weight'] = wts[u]
    adj = nx.adjacency_matrix(graph, nodelist=list(range(size)), weight=None)
    return graph, adj, wts


try:
    dqn_agent.load(model_origin)
except:
    print("Unable to load {}".format(model_origin))


baseline = mwds_vvv

best_IS_vec = []
loss_vec = []
results = pd.DataFrame([],
                       columns=["type", "size", "k", "p", "mwds", "mwcds", "gcn", "greedy", "ratio", "t0", "t1", "t2"])
csvname = "./output/{}_{}_test_foo.csv".format(model_origin.split('/')[-1], test_datapath.split('/')[-1])

epislon_reset = [5, 10, 15, 20]
epislon_val = 1.0
eval_size = FLAGS.nvalid
n_samples = FLAGS.ntrain
best_ratio = 3.0
last_ap = 1.0
batch_size = 100
tr_best = 0
for id in range(2000):
    losses = []
    losses_crt = []
    cnt = 0
    f_ct = 0
    size = np.random.choice([100, 150, 200, 250, 300])
    # size = np.random.choice([100, 150, 200, 250, 300, 350, 400, 450, 500])
    k = np.random.randint(10, 30)
    p = np.random.uniform(0.15, 0.35)
    # seed = id+epoch*100
    # graph, adj, wts = random_graph(size=size, k=k, p=p, gtype='grp', gseed=id+2000)
    graph, adj, wts = random_graph(size=size, k=k, p=p, gtype=FLAGS.gtype)
    if not nx.is_connected(graph):
        print("unconnected")
        continue
    adj_0 = adj.copy()
    nn = adj_0.shape[0]
    p_t = np.random.uniform(0.5, 0.9)
    # p_t = np.random.uniform(0.8, 0.9)
    terminals = nx.maximal_independent_set(graph)
    terminals = np.array(terminals)
    terminals = terminals[np.random.uniform(0.0, 1.0, size=(len(terminals, ))) <= p_t]
    # terminals = np.random.choice(wts.size, size=(int(np.round(0.1 * wts.size)),))
    terminals = set(terminals.tolist())
    # terminals = steiner_terminal_2hop(adj, wts)
    # mwcds_c, mwds_c, total_wt_c = baseline(adj, wts)
    newtime0 = time.time()
    mwcds_0, _ = heuristic_func(adj, wts, terminals)
    runtime0 = time.time() - newtime0
    solu_0 = mwcds_0 - terminals
    total_wt_0 = np.sum(wts[list(solu_0)])

    newtime1 = time.time()
    state, zs_t = dqn_agent.foo_train(adj_0, wts, terminals, train=True)
    runtime1 = time.time() - newtime1

    zs_np = zs_t.numpy()
    if dqn_agent.flags.diver_num == 2:
        gcn_wts = zs_np[:, 0].flatten() * wts.flatten() + zs_np[:, 1].flatten()
    else:
        gcn_wts = np.multiply(zs_np.flatten(), wts.flatten())
    top_wts = np.clip(gcn_wts, a_min=0.0, a_max=None)
    # top_wts = np.multiply(zs_t.numpy().flatten(), wts)
    mwcds_i, _ = heuristic_func(adj_0, top_wts, terminals)
    runtime2 = time.time() - newtime1

    solu_i = mwcds_i - terminals
    total_wt_i = np.sum(wts[list(solu_i)])
    p_ratio = total_wt_i/total_wt_0

    print("ID: {}".format(id),
          "gtype: GRP, size: {}, k: {}, p: {:.3f}".format(size, k, p),
          "Model: Actor",
          "steiner_i: {}".format(len(solu_i)),
          "steiner_0: {:.4f}".format(len(solu_0)),
          "gcn: {:.4f}".format(total_wt_i),
          "greedy: {:.4f}".format(total_wt_0),
          "ratio: {:.3f}".format(total_wt_i / total_wt_0),
          # "gcn ar: {:.3f}".format(total_wt_i / total_wt_c),
          # "grd ar: {:.3f}".format(total_wt_0 / total_wt_c),
          "runtimes: {:.2f}, {:.2f}, {:.2f}".format(runtime0, runtime1, runtime2),
          )
    results = results.append({"type": "GRP", "size": size, "k": k, "p": p,
                              "steiner_i": len(solu_i),
                              "steiner_0": len(solu_0),
                              "gcn": total_wt_i,
                              "greedy": total_wt_0,
                              # "vvv": total_wt_c,
                              # "ar_gb": total_wt_i/total_wt_c,
                              # "ar_db": total_wt_0/total_wt_c,
                              "ratio": total_wt_i/total_wt_0,
                              "t0": runtime0,
                              "t1": runtime1,
                              "t2": runtime2},
                              ignore_index=True)

    results.to_csv(csvname)
