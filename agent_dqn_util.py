# python3
# Make this standard template for testing and training
from __future__ import division
from __future__ import print_function

import sys
import os
import shutil
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
from natsort import natsorted, ns
sys.path.append( '%s/gcn' % os.path.dirname(os.path.realpath(__file__)) )
# add the libary path for graph reduction and local search
# sys.path.append( '%s/kernel' % os.path.dirname(os.path.realpath(__file__)) )
from gcn.models import GCN4_DQN
from gcn.utils import *
# import the libary for graph reduction and local search
# from reduce_lib import reducelib
import warnings
warnings.filterwarnings('ignore')
from runtime_config import flags, FLAGS
from heuristics import *

if not hasattr(flags.FLAGS, 'epsilon'):
    flags.DEFINE_float('epsilon', 1.0, 'initial exploration rate')
if not hasattr(flags.FLAGS, 'epsilon_min'):
    flags.DEFINE_float('epsilon_min', 0.001, 'minimal exploration rate')
if not hasattr(flags.FLAGS, 'epsilon_decay'):
    flags.DEFINE_float('epsilon_decay', 0.985, 'exploration rate decay per replay')
if not hasattr(flags.FLAGS, 'gamma'):
    flags.DEFINE_float('gamma', 1.0, 'gamma')

flags.DEFINE_float('actor_lr', 0.0005, 'test dataset')
flags.DEFINE_float('critic_lr', 0.001, 'test dataset')
flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_float('tau', 0.001, 'target network update')
flags.DEFINE_integer('train_start', 2000, 'train_start')

# Some preprocessing
num_supports = 1 + FLAGS.max_degree
model_func = GCN4_DQN
nsr = np.power(10.0, -FLAGS.snr_db/20.0)


args = flags.FLAGS


class Agent(object):
    """Distributed networked agents with shared trainable weights"""
    def __init__(self, input_flags, memory_size):
        self.feature_size = input_flags.feature_size
        self.memory = deque(maxlen=memory_size)
        self.reward_mem = deque(maxlen=memory_size)
        self.flags = input_flags
        self.placeholders = {
            'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=(None, self.flags.feature_size)),
            'hidden': tf.compat.v1.placeholder(tf.float32, shape=(None, self.flags.hidden1)),
            'adj': tf.compat.v1.sparse_placeholder(tf.float32),
            'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, self.flags.diver_num)),  # rewards
            'actions': tf.compat.v1.placeholder(tf.float32, shape=(None, self.flags.diver_num)),  # action space
            'labels_mask': tf.compat.v1.placeholder(tf.int32),
            'network_q': tf.compat.v1.placeholder(tf.float32, shape=()),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
            'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
        }
        self.delta = 0.000001  # prevent empty solution
        self.gamma = self.flags.gamma  # discount rate
        self.epsilon = self.flags.epsilon  # exploration rate
        self.epsilon_min = self.flags.epsilon_min
        self.epsilon_decay = self.flags.epsilon_decay
        self.learning_rate = self.flags.learning_rate
        self.sess = None
        self.hidden = None
        # self.writer = tf.summary.create_file_writer('./logs/metrics', max_queue=10000)
        self.saver = None

    def _build_model(self, name):
        raise NotImplementedError

    def makestate(self, adj, wts_nn):
        reduced_nn = wts_nn.shape[0]
        # norm_wts = np.amax(wts_nn) + 1e-9
        # norm_wts = np.amax(wts_nn, axis=0) + 1e-9
        # features = np.divide(wts_nn, norm_wts)
        norm_wts = 80000 # 100.0
        features = np.multiply(np.ones([reduced_nn, self.flags.feature_size]), wts_nn / norm_wts)
        features_raw = features.copy()
        features = sp.lil_matrix(features)
        features = sparse_to_tuple(features)
        support = simple_polynomials(adj, self.flags.max_degree)
        state = {"features": features, "support": support, "features_raw": features_raw, "adj": adj}
        return state

    def act(self, state, train):
        raise NotImplementedError

    def predict(self, state):
        raise NotImplementedError

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state.copy(), action.copy(), reward.copy(), next_state.copy(), done))

    def load(self, name):
        ckpt = tf.train.get_checkpoint_state(name)
        if ckpt:
            with self.sess.as_default():
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('loaded ' + ckpt.model_checkpoint_path)

    def save(self, name):
        with self.sess.as_default():
            self.saver.save(self.sess, os.path.join(name, "model.ckpt"))

    def copy_model_parameters(self, estimator1, estimator2):
        """
        Copies the model parameters of one estimator to another.
        Args:
          sess: Tensorflow session instance
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1)]
        e1_params = natsorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2)]
        e2_params = natsorted(e2_params, key=lambda v: v.name)

        update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            update_ops.append(op)

        self.sess.run(update_ops)

    def mellowmax(self, q_vec, omega, beta):
        c = np.max(q_vec)
        a_size = np.size(q_vec)
        mellow = c + np.log(np.sum(np.exp(omega * (q_vec - c))) / a_size) / omega
        # ans = np.sum(np.exp((q_vec-mellow)*beta)*(q_vec-mellow))
        return mellow

    def solve_mwis(self, adj_0, wts_0, train=False, grd=1.0):
        """
        GCN followed by LGS
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals = self.act(state, train)

        if self.flags.predict == 'mwis':
            # gcn_wts = np.divide(wts_nn.flatten(), act_vals.flatten()+1e-8)
            gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
            # gcn_wts = act_vals.flatten()+100
        else:
            gcn_wts = act_vals.flatten()
            # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
        # gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten()) + wts_nn.flatten()

        mwis, _ = local_greedy_search(adj, gcn_wts)
        # mwis, _ = greedy_search(adj, gcn_wts)
        solu = list(mwis)
        mwis_rt = mwis
        total_wt = np.sum(wts_nn[solu, 0])
        if train:
            # wts_norm = wts_nn[list(sol_gd), :]/greedy_util.flatten()
            # self.memorize(state.copy(), act_vals.copy(), list(sol_gd), wts_norm, 1.0)
            # reward = (total_wt + self.smallconst) / (greedy_util.flatten()[0] + self.smallconst)
            reward = total_wt / (grd + 1e-6)
            # reward = reward if reward > 0 else 0
            wts_norm = wts_nn/np.amax(wts_nn)
            if not np.isnan(reward):
                self.memorize(state.copy(), act_vals.copy(), list(mwis), {}, reward)
        return mwis_rt, total_wt


class A2CAgent(Agent):
    def __init__(self, input_flags, memory_size=5000):
        super(A2CAgent, self).__init__(input_flags, memory_size)
        # use gpu 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        # Initialize session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.target_update_iter = 10
        self.update_cnt = 0
        self.sess = tf.compat.v1.Session(config=config)
        self.model = self._build_model('model')
        self.target_model = self._build_model('target')
        self.action_dim = 1
        with self.sess.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
        # self.writer = tf.summary.create_file_writer('./logs/metrics', max_queue=10000)
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)

    def _build_model(self, name):
        # model = model_func(self.placeholders, flags=self.flags, name=name, logging=True)
        # model = model_func(self.placeholders, input_dim=1, name=name, logging=True)
        # Neural Net for Deep-Q learning Model
        model = model_func(self.placeholders,
                           hidden_dim=self.flags.hidden1,
                           num_layer=self.flags.num_layer,
                           # bias=True,
                           bias=False,
                           is_dual=False,
                           is_noisy=False,
                           # act=lambda x: x,
                           act=tf.nn.leaky_relu,
                           learning_rate=self.flags.learning_rate,
                           learning_decay=self.flags.learning_decay,
                           weight_decay=self.flags.weight_decay,
                           name=name,
                           logging=True)
        return model

    def predict(self, state):
        feed_dict_val = construct_feed_dict4pred(state["features"], state["support"],
                                                 self.placeholders, adj_coo=state["adj"])
        with self.sess.as_default():
            act_values, = self.sess.run([self.model.outputs], feed_dict=feed_dict_val)
        return act_values

    def act(self, state, train):
        act_values = self.predict(state)
        if train:
            if np.random.rand() <= self.epsilon:
                act_values = np.random.uniform(size=act_values.shape)
            # act_rand = np.random.uniform(0, 1.0, size=act_values.shape)
            # act_vals = np.random.uniform(0, 1.5, size=act_values.shape)
            # act_values = np.where(act_rand < self.epsilon, act_vals, act_values)
        return act_values  # returns action

    def update_target_model(self):
        """assign the current network parameters to target network"""
        # self.target_model.set_weights(self.model.get_weights())
        self.copy_model_parameters('model', 'target')
        self.update_cnt = 0

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None, None
        if self.update_cnt >= self.target_update_iter or self.update_cnt == 0:
            self.update_target_model()
        self.update_cnt += 1
        minibatch = random.sample(self.memory, batch_size)
        losses_act = []
        losses_crt = []
        states, targets_f, actions, hiddens_f = [], [], [], []
        for state, action, solu, next_state, reward in minibatch:
            # target = np.zeros_like(act_vals)
            target = 0
            # target[:, 0] = reward
            # target_f = np.zeros_like(action)
            # target_f = self.predict(state)
            target_f = action
            # target_f = target
            if next_state:
                feed_dict = construct_feed_dict(next_state['features'], next_state['support'], target_f,
                                                self.placeholders, adj_coo=state["adj"],
                                                actions=action, mask=1)
                val_next_state, = self.sess.run([self.model.outputs], feed_dict=feed_dict)
                target += reward + self.gamma * np.amax(val_next_state)
            else:
                target += reward
            target_f[solu, :] = target
            # target_f[:,0] = target
            # target_f[solu, :] = state['features_raw'][solu, :]
            states.append(state)
            targets_f.append(target_f)
            actions.append(action)
            # hiddens_f.append(hidden)

        for i in range(len(targets_f)):
            state = states[i]
            target_f = targets_f[i]
            act_vals = actions[i]
            # hidden = hiddens_f[i]
            feed_dict = construct_feed_dict(state['features'], state['support'], target_f, self.placeholders,
                                            adj_coo=state["adj"],
                                            actions=act_vals, mask=1)
            _, loss = self.sess.run([self.model.opt_op, self.model.loss], feed_dict=feed_dict)
            losses_crt.append(loss)

        # Keeping track of loss
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # with self.writer.as_default():
        #     tf.summary.scalar("critic loss", np.nanmean(losses_crt), step=self.step)
        #     tf.summary.scalar("actor loss", np.nanmean(losses_act), step=self.step)
        #     self.step += 1
        return np.nanmean(losses_act), np.nanmean(losses_crt)

    def utility(self, adj_0, wts_0, train=False):
        """
        GCN followed by LGS
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))
        # if self.hidden is None:
        #     self.hidden = np.zeros((wts_0.shape[0], self.flags.hidden1))
        # elif self.hidden.shape[0] != wts_0.shape[0]:
        #     self.hidden = np.zeros((wts_0.shape[0], self.flags.hidden1))

        state = self.makestate(adj, wts_nn)
        actions = self.act(state, train)
        # self.hidden = hidden

        return actions, state


