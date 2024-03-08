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

flags.DEFINE_string('gtype', 'er', 'training graph type: er, grp, ws, ba')
flags.DEFINE_string('test_datapath', './data/ER_Graph_Uniform_NP20_test', 'test dataset')
flags.DEFINE_integer('ntrain', 1, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('nvalid', 100, 'Number of outputs.')

from steiner_gcn_call_twin import DPGAgent, heuristic_func

# Get a list of available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Set the number of GPUs to use
num_gpus = len(gpus)
# Set up a MirroredStrategy to use all available GPUs
if num_gpus > 1:
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:%d" % i for i in range(num_gpus)])
else:
    strategy = tf.distribute.get_strategy() # default strategy
# Define and compile your model within the strategy scope
with strategy.scope():
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

# # use gpu 0
# os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
#
# # Initialize session
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True


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
    wts = np.random.uniform(0.00, 1.0, (size,))
    for u in graph:
        graph.nodes[u]['weight'] = wts[u]
    adj = nx.adjacency_matrix(graph, nodelist=list(range(size)), weight=None)
    return graph, adj, wts


try:
    dqn_agent.load(model_origin)
except:
    print("Unable to load {}".format(model_origin))

try:
    dqn_agent.load_critic(critic_origin)
except:
    print("Unable to load {}".format(model_origin))

test_len = 20
test_instances = []

gsizes = list(range(100, 350, 50))
sizes = np.random.choice(gsizes, size=(test_len,))
ks = np.random.randint(10, 30, size=(test_len,))
ps = np.random.uniform(0.15, 0.35, size=(test_len,))
j = 0
while len(test_instances) < test_len:
    graph, adj_0, wts_0 = random_graph(size=sizes[j], k=ks[j], p=ps[j], gtype=FLAGS.gtype)
    p_t = np.random.uniform(0.5, 0.9)
    terminals = nx.maximal_independent_set(graph)
    terminals = np.array(terminals)
    terminals = terminals[np.random.uniform(0.0, 1.0, size=(len(terminals, ))) <= p_t]
    terminals = set(terminals.tolist())
    test_instances.append((graph, adj_0, wts_0, terminals, sizes[j], ks[j], ps[j]))
    j += 1


best_IS_vec = []
loss_vec = []
results = pd.DataFrame([], columns=["data", "p"])
csvname = "./output/{}_{}_train_foo.csv".format(model_origin.split('/')[-1], test_datapath.split('/')[-1])

epislon_reset = [5, 10, 15, 20]
epislon_val = 1.0
eval_size = FLAGS.nvalid
n_samples = FLAGS.ntrain
best_ratio = 3.0
last_ap = 1.0
batch_size = 100
tr_best = 0
for epoch in range(FLAGS.epochs):
    losses = []
    losses_crt = []
    cnt = 0
    f_ct = 0
    p_corrs = []
    p_ratios = []
    z_means = []
    z_stds = []
    newtime = time.time()
    for id in np.random.permutation(2000):
        # size = np.random.choice([100, 150, 200, 250])
        # size = np.random.choice([100, 150, 200, 250, 300])
        size = np.random.choice(gsizes)
        k = np.random.randint(10, 30)
        p = np.random.uniform(0.15, 0.35)
        seed = id+epoch*100
        graph, adj, wts = random_graph(size=size, k=k, p=p, gtype=FLAGS.gtype)
        if not nx.is_connected(graph):
            print("unconnected")
            continue
        adj_0 = adj.copy()
        nn = adj_0.shape[0]
        p_t = np.random.uniform(0.5, 0.9)
        terminals = nx.maximal_independent_set(graph)
        terminals = np.array(terminals)
        terminals = terminals[np.random.uniform(0.0, 1.0, size=(len(terminals,))) <= p_t]
        terminals = set(terminals.tolist())

        mwcds_0, _ = heuristic_func(adj, wts, terminals)
        total_wt_0 = np.sum(wts[list(mwcds_0-terminals)])

        state, zs_t = dqn_agent.foo_train(adj_0, wts, terminals, train=True)
        zs_np = zs_t.numpy()
        if dqn_agent.flags.diver_num == 2:
            gcn_wts = zs_np[:, 0].flatten() * wts.flatten() + zs_np[:, 1].flatten()
        else:
            gcn_wts = np.multiply(zs_np.flatten(), wts.flatten())
        top_wts = np.clip(gcn_wts, a_min=0.0, a_max=None)
        mwcds_i, _ = heuristic_func(adj_0, top_wts, terminals)
        total_wt_i = np.sum(wts[list(mwcds_i-terminals)])
        ind_vec, apu_avg = dqn_agent.predict_train(adj_0, terminals, zs_t, state, n_samples=n_samples)
        p_ratio = total_wt_i/total_wt_0
        p_ratios.append(p_ratio)
        p_corrs.append(apu_avg)
        z_means.append(np.mean(zs_t.numpy()))
        z_stds.append(np.std(zs_t.numpy()))
        f_ct += 1
        if cnt < batch_size - 1:
            cnt += 1
            continue
        else:
            cnt = 0
            runtime = time.time() - newtime
            newtime = time.time()

        test_ratio = []
        test_ratio2 = []
        for j in range(test_len):
            graph, adj_1, wts_1, terminals, nn, k, p = test_instances[j]
            state, zs_t = dqn_agent.foo_train(adj_1, wts_1, terminals, train=False)
            zs_np = zs_t.numpy()
            if dqn_agent.flags.diver_num == 2:
                gcn_wts = zs_np[:, 0].flatten() * wts_1.flatten() + zs_np[:, 1].flatten()
            else:
                gcn_wts = np.multiply(zs_np.flatten(), wts_1.flatten())
            top_wts = np.clip(gcn_wts, a_min=0.0, a_max=None)
            mwcds_0, _ = heuristic_func(adj_1, wts_1, terminals)
            total_wt_0 = np.sum(wts_1[list(mwcds_0 - terminals)])
            mwcds_i, _ = heuristic_func(adj_1, top_wts, terminals)
            total_wt_i = np.sum(wts_1[list(mwcds_i - terminals)])
            test_ratio.append(total_wt_i / total_wt_0)

        if np.mean(test_ratio) < best_ratio:
            dqn_agent.save(os.path.join(model_origin, 'cp-{epoch:04d}.ckpt'.format(epoch=epoch)))
            dqn_agent.save_critic(os.path.join(critic_origin, 'cp-{epoch:04d}.ckpt'.format(epoch=epoch)))
            best_ratio = np.mean(test_ratio)

        loss = dqn_agent.replay(batch_size)
        losses.append(loss)
        loss_crt = dqn_agent.replay_crt(batch_size)
        losses_crt.append(loss_crt)

        tr_factor = -np.nanmean(test_ratio)/loss
        if tr_factor > tr_best:
            tr_best = tr_factor
        tr_dive = (tr_factor - tr_best)/tr_best

        print("Epoch: {}".format(epoch),
              "ID: %03d" % f_ct,
              "Model: Actor",
              "Train_Ratio: {:.4f}".format(np.mean(p_ratios)),
              "Test_Ratio: {:.4f}".format(np.mean(test_ratio)),
              "Loss: {:.4f}".format(loss),
              "Corr: {:.4f}".format(np.mean(p_corrs)),
              "L_crt: {:.4f}".format(loss_crt),
              "Track: {:.4f}".format(tr_factor),
              "runtime: {:.2f}".format(runtime),
              "z_std: {:.3f}".format(np.nanmean(z_stds)),
              "z_avg: {:.3f}".format(np.nanmean(z_means)),
              flush=True
              )
        p_ratios = []
        z_means = []
        z_stds = []
        p_corrs = []

    loss_vec.append(np.mean(losses))
print(loss_vec)

