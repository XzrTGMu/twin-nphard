# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )

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

from mwds_gcn_call_twin import DPGAgent, heuristic_func
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
        graph = nx.generators.random_graphs.fast_gnp_random_graph(size, float(k)/float(size))
    elif gtype == 'ba':
        graph = nx.generators.random_graphs.barabasi_albert_graph(size, int(np.round(k * p)))
    else:
        raise ValueError('Unsupported graph type')
    wts = np.random.uniform(0.01, 1.0, (size,))
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
results = pd.DataFrame([], columns=["type", "size", "k", "p", "mwds", "mwcds", "gcn", "greedy", "ratio",
                                    "t0","t1","t2","t3"])
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
    # size = np.random.choice([100, 150, 200, 250, 300, 350])
    size = np.random.choice([100, 150, 200, 250, 300, 350, 400, 450, 500])
    k = np.random.randint(10, 30)
    p = np.random.uniform(0.15, 0.35)
    # seed = id+epoch*100
    graph, adj, wts = random_graph(size=size, k=k, p=p, gtype=FLAGS.gtype, gseed=id+2000)
    adj_0 = adj.copy()
    nn = adj_0.shape[0]
    newtime0 = time.time()
    mwds_c, gray_c, total_wt_c = baseline(adj, wts)
    runtime0 = time.time() - newtime0
    newtime1 = time.time()
    mwds_0, gray_0, total_wt_0 = heuristic_func(adj, wts)
    runtime1 = time.time() - newtime1

    newtime2 = time.time()
    state, zs_t = dqn_agent.foo_train(adj_0, wts, train=True)
    runtime2 = time.time() - newtime2
    zs_np = zs_t.numpy()
    if dqn_agent.flags.diver_num == 2:
        gcn_wts = zs_np[:, 0].flatten() * wts.flatten() + zs_np[:, 1].flatten()
    else:
        gcn_wts = np.multiply(zs_np.flatten(), wts.flatten())
    top_wts = np.clip(gcn_wts, a_min=0.0, a_max=None)
    mwds_i, gray_i, _ = heuristic_func(adj_0, top_wts)
    runtime3 = time.time() - newtime2
    total_wt_i = np.sum(wts[list(mwds_i)])
    p_ratio = total_wt_i/total_wt_0

    assert nx.is_dominating_set(graph, mwds_i)
    assert nx.is_dominating_set(graph, mwds_c)
    assert nx.is_dominating_set(graph, mwds_0)

    print("ID: {}".format(id),
          "gtype: {}, size: {}, k: {}, p: {:.3f}".format(FLAGS.gtype, size, k, p),
          "Model: Actor",
          "mwds_i: {}".format(len(mwds_i)),
          "mwds_0: {:.4f}".format(len(mwds_0)),
          "gcn: {:.4f}".format(total_wt_i),
          "greedy: {:.4f}".format(total_wt_0),
          "ratio: {:.3f}".format(total_wt_i / total_wt_0),
          "gcn ar: {:.3f}".format(total_wt_i / total_wt_c),
          "grd ar: {:.3f}".format(total_wt_0 / total_wt_c),
          "runtime: {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(runtime0, runtime1, runtime2, runtime3),
          )
    results = results.append({"type": FLAGS.gtype, "size": size, "k": k, "p": p,
                              "mwds_i": len(mwds_i),
                              "mwds_0": len(mwds_0),
                              "gcn": total_wt_i,
                              "greedy": total_wt_0,
                              "vvv": total_wt_c,
                              "ar_gb": total_wt_i/total_wt_c,
                              "ar_db": total_wt_0/total_wt_c,
                              "ratio": total_wt_i/total_wt_0,
                              "t0": runtime0,
                              "t1": runtime1,
                              "t2": runtime2,
                              "t3": runtime3},
                              ignore_index=True)

    results.to_csv(csvname)
