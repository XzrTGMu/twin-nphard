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

import tensorflow as tf
from collections import deque
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
# Settings (FLAGS)
from runtime_config import *
from heuristics import *

flags.DEFINE_integer('opt', 1, '1: dgcn_lgs, 2: dgcn_cgs, 3: gcn_rollout')

from mwis_gcn_call_twin import DQNAgent # Twin
dqn_agent = DQNAgent(FLAGS, 5000)

# test data path
test_datapath = FLAGS.datapath
test_mat_names = sorted(os.listdir(test_datapath))

# Some preprocessing
noout = min(FLAGS.diver_num, FLAGS.diver_out) # number of outputs
time_limit = FLAGS.timeout  # time limit for searching
backoff_thresh = 1 - FLAGS.backoff_prob

num_supports = 1 + FLAGS.max_degree
nsr = np.power(10.0, -FLAGS.snr_db/20.0)

from directory import create_result_folder, find_model_folder
model_origin = find_model_folder(FLAGS, 'dqn')


def weighted_random_graph(N, p, dist, maxWts=1.0):
    graph = nx.generators.random_graphs.fast_gnp_random_graph(N,p)
    if dist.lower() == 'uniform':
        for u in graph:
            graph.nodes[u]['weight'] = np.random.uniform(0,maxWts)
    elif dist.lower() == 'normal_l1':
        for u in graph:
            graph.nodes[u]['weight'] = np.abs(np.random.randn())
    elif dist.lower() == 'normal_l2':
        for u in graph:
            graph.nodes[u]['weight'] = np.square(np.random.randn())

    return graph

# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

try:
    dqn_agent.load(model_origin)
except:
    print("Unable to load {}".format(model_origin))

best_IS_vec = []
loss_vec = []

epislon_reset = [5, 10, 15, 20]
epislon_val = 1.0

best_ratio = 0.95
results = pd.DataFrame([], columns=["data", "p", "step_lgs", "step_gcn","t0","t1","t2"])
p_ratios = []
postfix = 'lgs'
# csvname = "./output/{}_{}_{}.csv".format(model_origin.split('/')[-1], test_datapath.split('/')[-1], postfix)
csvname = "./output/{}_{}_{}_local_complexity.csv".format(model_origin.split('/')[-1], test_datapath.split('/')[-1], postfix)

for id in range(len(test_mat_names)):
    best_IS_num = -1
    mat_contents = sio.loadmat(test_datapath + '/' + test_mat_names[id])
    adj_0 = mat_contents['adj']
    wts = mat_contents['weights'].transpose()
    # _, greedy_util = local_greedy_search(adj_0, wts)
    newtime0 = time.time()
    _, greedy_util, step_gdy = local_greedy_search_stats(adj_0, wts)
    runtime0 = time.time()-newtime0
    nn = adj_0.shape[0]
    bsf_q = []
    q_ct = 0
    res_ct = 0
    out_id = -1

    newtime1 = time.time()
    act_vals, _ = dqn_agent.utility(adj_0, wts, train=False)
    runtime1 = time.time() - newtime1
    gcn_wts = np.multiply(act_vals.flatten(), wts.flatten())
    # mwis, _ = local_greedy_search(adj_0, gcn_wts)
    mwis, _, step_gcn = local_greedy_search_stats(adj_0, gcn_wts)
    runtime2 = time.time() - newtime1
    ss_util = np.sum(wts[list(mwis)])


    p_ratio = ss_util.flatten()/greedy_util.flatten()
    p_ratios.append(p_ratio[0])
    test_ratio = []
    print("ID: %03d" % id,
          "File: {}".format(test_mat_names[id]),
          "Ratio: {:.6f}".format(p_ratio[0]),
          "Avg_Ratio: {:.6f}".format(np.mean(p_ratios)),
          # "Avg_IS_Size: {:.4f}".format(avg_is_size),
          "runtime: {:.3f}, {:.3f}, {:.3f}".format(runtime0, runtime1, runtime2))

    # results = results.append({"data": test_mat_names[id],
    #                           "p": p_ratio[0],
    #                           },
    #                          ignore_index=True)
    results = results.append({"data": test_mat_names[id],
                              "p": p_ratio[0],
                              "step_lgs": step_gdy,
                              "step_gcn": step_gcn,
                              "t0": runtime0,
                              "t1": runtime1,
                              "t2": runtime2
                              },
                             ignore_index=True)

results.to_csv(csvname)
