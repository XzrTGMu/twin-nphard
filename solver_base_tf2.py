# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil

sys.path.append('%s/gcn' % os.path.dirname(os.path.realpath(__file__)))
import time
import random
import scipy.io as sio
import numpy as np
import scipy.sparse as sp
from multiprocessing import Queue
from copy import deepcopy
import networkx as nx

import tensorflow as tf
from collections import deque
from gcn.utils import *
import warnings

warnings.filterwarnings('ignore')

from runtime_config import flags, FLAGS
# Settings (FLAGS)
from heuristics_mwcds import *

if not hasattr(flags.FLAGS, 'epsilon'):
    flags.DEFINE_float('epsilon', 1.0, 'initial exploration rate')
if not hasattr(flags.FLAGS, 'epsilon_min'):
    flags.DEFINE_float('epsilon_min', 0.001, 'minimal exploration rate')
if not hasattr(flags.FLAGS, 'epsilon_decay'):
    flags.DEFINE_float('epsilon_decay', 0.985, 'exploration rate decay per replay')
if not hasattr(flags.FLAGS, 'gamma'):
    flags.DEFINE_float('gamma', 1.0, 'gamma')

# Some preprocessing
num_supports = 1 + FLAGS.max_degree
nsr = np.power(10.0, -FLAGS.snr_db / 20.0)


class Solver(object):
    def __init__(self, input_flags, memory_size):
        self.feature_size = input_flags.feature_size
        self.memory = deque(maxlen=memory_size)
        self.reward_mem = deque(maxlen=memory_size)
        self.flags = input_flags
        self.delta = 0.000001  # prevent empty solution
        self.gamma = self.flags.gamma  # discount rate
        self.epsilon = self.flags.epsilon  # exploration rate
        self.epsilon_min = self.flags.epsilon_min
        self.epsilon_decay = self.flags.epsilon_decay
        self.learning_rate = self.flags.learning_rate
        self.sess = None
        self.saver = None

    def _build_model(self):
        raise NotImplementedError

    def makestate(self, adj, wts_nn):
        reduced_nn = wts_nn.shape[0]
        norm_wts = np.amax(wts_nn) + 1e-9
        if self.flags.predict == 'mwis':
            features = np.ones([reduced_nn, self.flags.feature_size])
        else:
            features = np.multiply(np.ones([reduced_nn, self.flags.feature_size]), wts_nn / norm_wts)
        features_raw = features.copy()
        features = sp.lil_matrix(features)
        if self.flags.predict == 'mwis':
            features = preprocess_features(features)
        else:
            features = sparse_to_tuple(features)
        support = simple_polynomials(adj, self.flags.max_degree)
        state = {"features": features, "support": support, "features_raw": features_raw}
        return state

    def act(self, state, train):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def memorize(self, state, act_vals, solu, next_state, reward):
        self.memory.append((state.copy(), act_vals.copy(), solu.copy(), next_state.copy(), reward))
        self.reward_mem.append(reward)

    def load(self, name):
        ckpt = tf.train.get_checkpoint_state(name)
        if ckpt:
            with self.sess.as_default():
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('loaded ' + ckpt.model_checkpoint_path)

    def save(self, name):
        with self.sess.as_default():
            self.saver.save(self.sess, os.path.join(name, "model.ckpt"))

    def mellowmax(self, q_vec, omega, beta):
        c = np.max(q_vec)
        a_size = np.size(q_vec)
        mellow = c + np.log(np.sum(np.exp(omega * (q_vec - c))) / a_size) / omega
        return mellow

    def utility(self, adj_0, wts_0, train=False):
        """
        GCN for per utility function
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals = self.act(state, train)

        gcn_wts = act_vals.numpy()

        return gcn_wts, state


# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

