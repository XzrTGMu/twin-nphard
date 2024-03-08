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
import networkx as nx
from scipy.stats.stats import pearsonr

import tensorflow as tf
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
from runtime_config import flags, FLAGS
# Settings (FLAGS)
from heuristics_mwcds import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD, schedules
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import ChebConv
from spektral.transforms import LayerPreprocess
from solver_base_tf2 import Solver
tf.config.run_functions_eagerly(True)

# Some preprocessing
nsr = np.power(10.0, -FLAGS.snr_db/20.0)

# heuristic_func = node_weight_steiner_set
heuristic_func = nwst_sph


class DPGAgent(Solver):
    def __init__(self, input_flags, memory_size=5000):
        super(DPGAgent, self).__init__(input_flags, memory_size)
        self.flags = input_flags
        self.num_supports = 1 + self.flags.max_degree
        self.l2_reg = 5e-4
        self.model = self._build_model()
        self.memory_crt = deque(maxlen=memory_size)
        self.mse = MeanSquaredError()
        self.critic = self._build_critic(num_layer=5)
        self.critic.trainable = True
        self.model.trainable = True

    def _build_model(self):
        # Neural Net for Actor Model
        x_in = Input(shape=(self.feature_size+2,), dtype=tf.float64, name="x_in")
        a_in = Input((None, ), sparse=True, dtype=tf.float64, name="a_in")

        gc_l = x_in
        for l in range(self.flags.num_layer):
            if l < self.flags.num_layer - 1:
                act = "leaky_relu"
                output_dim = self.flags.hidden1
            else:
                output_dim = self.flags.diver_num
                if output_dim == 1:
                    act = "sigmoid"
                elif output_dim == 2:
                    act = "tanh"
                else:
                    raise ValueError("unsupported diver_num {}".format(output_dim))

            do_l = Dropout(self.flags.dropout, dtype='float64')(gc_l)
            gc_l = ChebConv(
                output_dim, K=self.num_supports, activation=act,
                kernel_regularizer=l2(self.l2_reg),
                use_bias=True,
                dtype='float64'
            )([do_l, a_in])

        # gc_l = (gc_l - tf.reduce_mean(gc_l))/(6*tf.math.reduce_std(gc_l)) + 0.5
        # gc_l = tf.abs(gc_l)
        # gc_l = (gc_l - tf.reduce_mean(gc_l)) + 0.5
        # Build model
        model = Model(inputs=[x_in, a_in], outputs=gc_l)
        if self.flags.learning_decay == 1.0:
            self.optimizer = Adam(learning_rate=self.learning_rate)
        else:
            lr_schedule = schedules.ExponentialDecay(
                initial_learning_rate=self.learning_rate,
                decay_steps=200,
                decay_rate=self.flags.learning_decay)
            self.optimizer = Adam(learning_rate=lr_schedule)
        model.summary()
        return model

    def _build_critic(self, num_layer=3, dropout=0.0):
        # Neural Net for Critic Model
        x_in = Input(shape=(self.flags.diver_num+3,), dtype=tf.float64, name="x_in")
        a_in = Input((None, ), sparse=True, dtype=tf.float64, name="a_in")

        # gc_l = (x_in - tf.reduce_mean(x_in))/(6*tf.math.reduce_std(x_in)) + 0.5
        # gc_l = x_in - tf.reduce_mean(x_in) # + 0.5
        # x_in_f = x_in[:, -2:-1] - tf.reduce_mean(x_in[:, -2:-1]) + 0.5
        # gc_l = tf.concat([x_in[:, 0:-1], x_in_f], axis=-1)
        gc_l = x_in
        for l in range(num_layer):
            if l < num_layer - 1:
                act = "leaky_relu"
                output_dim = 32
            else:
                act = "softmax"
                output_dim = 2
            do_l = Dropout(dropout, dtype='float64')(gc_l)
            gc_l = ChebConv(
                output_dim, K=self.num_supports, activation=act,
                kernel_regularizer=l2(self.l2_reg),
                use_bias=True,
                dtype='float64'
            )([do_l, a_in])

        # Build model
        model = Model(inputs=[x_in, a_in], outputs=gc_l)
        if self.flags.learning_decay == 1.0:
            self.opt_crt = Adam(learning_rate=0.0001)
        else:
            lr_schedule = schedules.ExponentialDecay(
                initial_learning_rate=0.0001,
                decay_steps=200,
                decay_rate=self.flags.learning_decay)
            self.opt_crt = Adam(learning_rate=lr_schedule)
        model.summary()
        return model

    def load_critic(self, name):
        ckpt = tf.train.latest_checkpoint(name)
        if ckpt:
            self.critic.load_weights(ckpt)
            print('Critic loaded ' + ckpt)

    def load(self, name):
        ckpt = tf.train.latest_checkpoint(name)
        if ckpt:
            self.model.load_weights(ckpt)
            print('Actor loaded ' + ckpt)

    def save(self, checkpoint_path):
        self.model.save_weights(checkpoint_path)

    def save_critic(self, checkpoint_path):
        self.critic.save_weights(checkpoint_path)

    def makestate(self, adj, wts_nn):
        reduced_nn = wts_nn.shape[0]
        # features = np.ones([reduced_nn, self.feature_size])
        features = np.multiply(np.ones([reduced_nn, self.feature_size]), wts_nn)
        support = simple_polynomials(adj, self.flags.max_degree)
        state = {"features": features, "support": support[1]}
        return state

    def memorize_crt(self, grad, loss, reward):
        self.memory_crt.append((grad.copy(), loss, reward))

    def predict(self, state):
        x_in = tf.convert_to_tensor(state["features"], dtype=tf.float64)
        coord, values, shape = state["support"]
        a_in = tf.sparse.SparseTensor(coord, values, shape)
        act_values = self.model([x_in, a_in])
        return act_values, np.argmax(act_values.numpy())

    def act(self, state, train, explore=0.0):
        act_values, action = self.predict(state)
        if explore > 0.001:
            noise = tf.random.uniform(act_values.shape, -explore, explore, dtype=act_values.dtype)
            act_values += noise
        return act_values, action  # returns action

    def predict_critic(self, z_out, state):
        x_in = tf.convert_to_tensor(state["features"], dtype=tf.float64)
        coord, values, shape = state["support"]
        a_in = tf.sparse.SparseTensor(coord, values, shape)
        x_in = tf.concat([x_in, z_out], axis=1)
        sch_pred = self.critic([x_in, a_in])
        return sch_pred

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return float('NaN')
        self.reward_mem.clear()
        minibatch = random.sample(self.memory, batch_size)
        losses = []
        for grad, _, _, loss, _ in minibatch:
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
            losses.append(loss)

        self.memory.clear()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return np.nanmean(losses)

    def replay_crt(self, batch_size):
        if len(self.memory_crt) < batch_size:
            return float('NaN')
        minibatch = random.sample(self.memory_crt, batch_size)
        losses = []
        for grad, loss, _ in minibatch:
            self.opt_crt.apply_gradients(zip(grad, self.critic.trainable_weights))
            losses.append(loss)

        self.memory_crt.clear()
        return np.nanmean(losses)

    def foo_train(self, adj_0, wts_0, terminals, train=False):
        adj = adj_0.copy()
        nn  = wts_0.shape[0]
        wts_nn = np.reshape(wts_0, (nn, FLAGS.feature_size))
        gsignal = np.ones(shape=(nn, FLAGS.feature_size + 2), dtype=np.float)
        # one hot encoding of the terminals
        term_list = list(terminals)
        gsignal[:, 1] = 0
        gsignal[term_list, 1] = 1
        gsignal[term_list, 2] = 0

        # GCN
        with tf.GradientTape() as g:
            g.watch(self.model.trainable_weights)
            state = self.makestate(adj, gsignal)
            act_val, act = self.act(state, train, explore=0.0)
            act_val_norm = act_val
            # act_val_norm = 0.5 + (act_val - tf.reduce_mean(act_val))
            sch_pred = self.predict_critic(act_val_norm, state)
            regularization_loss = tf.reduce_sum(self.model.losses)
            obj_fn = tf.reduce_sum(sch_pred[:, 0])
            obj_fn = obj_fn + regularization_loss
        if train:
            gradients = g.gradient(obj_fn, self.model.trainable_weights)
            self.memorize(gradients, [], [], obj_fn.numpy(), 0)
        return state, act_val

    def predict_train(self, adj, terminals, zs_0, state, n_samples=1, z_std=0.15):
        """
        GCN followed by LGS
        wts_0: topology weighted utility
        """
        zs_nn = zs_0.numpy()
        nn = zs_nn.shape[0]
        adj_0 = adj.copy()

        # GCN
        with tf.GradientTape() as g:
            g.watch(self.critic.trainable_weights)
            sch_pred = self.predict_critic(zs_0, state)

            ind_vec = np.zeros((nn, 2))
            apu_avg = 0.0
            for i in range(n_samples):
                wts = np.random.uniform(0.00, 1.0, size=(nn, 1))
                # zs_i = zs_nn + np.random.normal(0, z_std, size=(nn, 1))
                # zs_i = zs_nn
                zs_i = zs_nn + np.random.uniform(-0.5*z_std, 0.5*z_std, size=(nn, self.flags.diver_num))
                # zs_i = np.clip(zs_i, a_min=0.0, a_max=None)
                if FLAGS.predict == 'mpy':
                    if self.flags.diver_num == 2:
                        gcn_wts = zs_i[:, 0].flatten() * wts.flatten() + zs_i[:, 1].flatten()
                    else:
                        gcn_wts = np.multiply(zs_i.flatten(), wts.flatten())
                    gcn_wts = np.clip(gcn_wts, a_min=0.0, a_max=None)
                    # gcn_wts = np.multiply(zs_i.flatten(), wts.flatten())
                else:
                    gcn_wts = zs_i.flatten()
                # gcn_wts = zs_i.flatten()
                mwcds_i, total_wt_i = heuristic_func(adj_0, gcn_wts, terminals)
                solu = list(mwcds_i)
                ind_vec[solu, 0] += wts[solu, 0]/(float(n_samples))

            # reward = apu_avg
            ind_vec[:, 1] = 1.0 - ind_vec[:, 0]
            corr, pval = pearsonr(ind_vec[:, 0], sch_pred[:, 0].numpy())
            reward = corr
            y_target = tf.convert_to_tensor(ind_vec, dtype=tf.float64)
            regularization_loss = tf.reduce_sum(self.critic.losses)
            loss_value = tf.sqrt(self.mse(y_target, sch_pred)) + regularization_loss
        gradients = g.gradient(loss_value, self.critic.trainable_weights)
        self.memorize_crt(gradients, loss_value.numpy(), reward)
        return ind_vec, reward

