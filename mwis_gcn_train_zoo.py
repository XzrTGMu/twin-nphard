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

import tensorflow as tf
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
# Settings (FLAGS)
from runtime_config import flags, FLAGS
from heuristics import *

flags.DEFINE_string('test_datapath', './data/ER_Graph_Uniform_NP20_test', 'test dataset')
flags.DEFINE_integer('ntrain', 1, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('nvalid', 100, 'Number of outputs.')
from mwis_gcn_call_twin import DQNAgent
dqn_agent = DQNAgent(FLAGS, 5000)

# test data path
data_path = FLAGS.datapath
test_datapath = FLAGS.test_datapath
val_mat_names = sorted(os.listdir(data_path))
test_mat_names = sorted(os.listdir(test_datapath))

# Some preprocessing
noout = min(FLAGS.diver_num, FLAGS.diver_out) # number of outputs
time_limit = FLAGS.timeout  # time limit for searching
backoff_thresh = 1 - FLAGS.backoff_prob

num_supports = 1 + FLAGS.max_degree
nsr = np.power(10.0, -FLAGS.snr_db/20.0)

from directory import create_result_folder, find_model_folder
model_origin = find_model_folder(FLAGS, 'dqn')

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
results = pd.DataFrame([], columns=["data", "p"])
csvname = "./output/{}_{}_train_zoo.csv".format(model_origin.split('/')[-1], test_datapath.split('/')[-1])

epislon_reset = [5, 10, 15, 20]
epislon_val = 1.0
eval_size = FLAGS.nvalid
n_samples = FLAGS.ntrain
best_ratio = 1.0
last_ap = 1.0
batch_size = 100
zoo_num_points = flags.FLAGS.ntrain
zoo_mu = 0.15
tr_best = 0
for epoch in range(FLAGS.epochs):
    losses = []
    losses_crt = []
    cnt = 0
    f_ct = 0
    q_totals = []
    p_ratios = []
    z_means = []
    newtime = time.time()
    for id in np.random.permutation(len(val_mat_names)):
        best_IS_num = -1
        mat_contents = sio.loadmat(data_path + '/' + val_mat_names[id])
        adj_0 = mat_contents['adj']
        nn = adj_0.shape[0]
        wts = np.random.uniform(0, 1, size=(nn, 1))
        start_time = time.time()
        _, greedy_util = greedy_search(adj_0, wts)
        # GCN
        with tf.GradientTape(persistent=True) as g:
            g.watch(dqn_agent.model.trainable_weights)
            state = dqn_agent.makestate(adj_0, wts)
            Z_mtx, fast_params = dqn_agent.act(state, train=False)
            g.watch(Z_mtx)
            act_vals = Z_mtx.numpy()
            n_loop = zoo_num_points + 1
            grad = [Z_mtx]
            grad_np = [grad_ts.numpy() for grad_ts in grad]
            grad_wts = [np.zeros_like(grad_wi) for grad_wi in grad_np]
            for i in range(n_loop):
                i_zs = deepcopy(grad_np)
                if i:
                    mu_i = [np.random.normal(0, 1.0, act_vals.shape)]
                    # mu_i = [np.random.uniform(-1.0, 1.0, act_vals.shape)]
                    for wi in range(len(grad)):
                        i_zs[wi] += mu_i[wi] * zoo_mu
                else:
                    mu_i = [np.zeros_like(wts) for wts in grad_np]

                if dqn_agent.flags.predict == 'mwis':
                    gcn_wts = np.multiply(i_zs[0].flatten(), wts.flatten())
                else:
                    gcn_wts = i_zs[0].flatten()
                mwis, _ = local_greedy_search(adj_0, gcn_wts)
                solu = list(mwis)
                ss_util = np.sum(wts[solu, 0])
                opt_obj = ss_util / (greedy_util + 1e-6)
                if i == 0:
                    opt_obj_0 = opt_obj
                else:
                    for wi in range(len(grad_np)):
                        grad_diff = - (opt_obj - opt_obj_0) * mu_i[wi] # maximize
                        grad_wts[wi] += grad_diff/zoo_mu

        grad_z = [tf.convert_to_tensor(np.clip(grad_wts_j, -10.0, 10.0)) for grad_wts_j in grad_wts]
        grad_gcn = g.gradient(grad, dqn_agent.model.trainable_weights, output_gradients=grad_z)
        regularization_loss = tf.reduce_sum(dqn_agent.model.losses)
        grad_reg = g.gradient(regularization_loss, dqn_agent.model.trainable_weights)
        for ii in range(len(grad_gcn)):
            if grad_reg[ii] is not None:
                grad_gcn[ii] += grad_reg[ii]
        dqn_agent.memorize(grad_gcn, [], [], opt_obj_0, None)
        del g

        p_ratio = opt_obj_0
        solu = list(mwis)
        q_totals.append(len(solu))
        p_ratios.append(p_ratio)
        z_means.append(np.mean(act_vals))
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
        test_len = len(test_mat_names)
        for j in range(test_len):
            mat_contents = sio.loadmat(test_datapath + '/' + test_mat_names[j % test_len])
            adj_0 = mat_contents['adj']
            wts = mat_contents['weights'].transpose()
            nn = adj_0.shape[0]
            _, greedy_util = greedy_search(adj_0, wts)
            bsf_q = []
            q_ct = 0
            res_ct = 0
            out_id = -1
            _, best_IS_util = dqn_agent.solve_mwis(adj_0, wts, train=False)
            test_ratio.append(best_IS_util / greedy_util)

        if np.mean(test_ratio) > best_ratio:
            dqn_agent.save(os.path.join(model_origin, 'cp-{epoch:04d}.ckpt'.format(epoch=epoch)))
            best_ratio = np.mean(test_ratio)

        loss_crt = float('NaN')
        loss = dqn_agent.replay(batch_size)
        if loss is None:
            loss = float('NaN')
        losses.append(loss)

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
              "L_Avg: {:.4f}".format(np.mean(losses)),
              "Track: {:.4f}".format(tr_factor),
              "runtime: {:.2f}".format(runtime),
              "z_avg: {:.3f}".format(np.nanmean(z_means)))
        p_ratios = []
        z_means = []

    loss_vec.append(np.mean(losses))

print(loss_vec)

