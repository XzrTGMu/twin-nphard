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

import tensorflow as tf
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from gcn.utils import *
from runtime_config import flags, FLAGS
# Settings (FLAGS)
from heuristics import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import ChebConv
from spektral.transforms import LayerPreprocess
from solver_base_tf2 import Solver
tf.config.run_functions_eagerly(True)

# Some preprocessing
nsr = np.power(10.0, -FLAGS.snr_db/20.0)


class GDPGAgent(Solver):
    def __init__(self, input_flags, memory_size=5000):
        super(GDPGAgent, self).__init__(input_flags, memory_size)
        self.flags = input_flags
        self.num_supports = 1 + self.flags.max_degree
        self.l2_reg = 5e-4
        self.model = self._build_model()
        self.memory_crt = deque(maxlen=memory_size)
        self.memory_act = deque(maxlen=memory_size)
        self.mse = MeanSquaredError()
        self.critic = self._build_critic(num_layer=3)
        self.critic.trainable = False

    def _build_model(self):
        # Neural Net for Actor Model
        x_in = Input(shape=(self.feature_size,), dtype=tf.float64, name="x_in")
        a_in = Input((None, ), sparse=True, dtype=tf.float64, name="a_in")

        gc_l = x_in
        for l in range(self.flags.num_layer):
            if l < self.flags.num_layer - 1:
                act = "leaky_relu"
                output_dim = self.flags.hidden1
            else:
                act = "relu"
                output_dim = self.flags.diver_num
            do_l = Dropout(self.flags.dropout, dtype='float64')(gc_l)
            gc_l = ChebConv(
                output_dim, K=self.num_supports, activation=act,
                kernel_regularizer=l2(self.l2_reg),
                use_bias=False,
                dtype='float64'
            )([do_l, a_in])

        # gc_l = (gc_l - tf.reduce_mean(gc_l))/(6*tf.math.reduce_std(gc_l)) + 0.5
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
        x_in = Input(shape=(self.flags.diver_num,), dtype=tf.float64, name="x_in")
        a_in = Input((None, ), sparse=True, dtype=tf.float64, name="a_in")

        # gc_l = (x_in - tf.reduce_mean(x_in))/(6*tf.math.reduce_std(x_in)) + 0.5
        gc_l = x_in
        for l in range(num_layer):
            if l < num_layer - 1:
                act = "leaky_relu"
                output_dim = 32
            else:
                act = "linear"
                output_dim = 1
            do_l = Dropout(dropout, dtype='float64')(gc_l)
            gc_l = ChebConv(
                output_dim, K=self.num_supports, activation=act,
                kernel_regularizer=l2(self.l2_reg),
                use_bias=False,
                dtype='float64'
            )([do_l, a_in])

        # Build model
        model = Model(inputs=[x_in, a_in], outputs=gc_l)
        if self.flags.learning_decay == 1.0:
            self.opt_crt = Adam(learning_rate=0.001)
        else:
            lr_schedule = schedules.ExponentialDecay(
                initial_learning_rate=0.001,
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

    def save_critic(self, checkpoint_path):
        self.critic.save_weights(checkpoint_path)

    def load(self, name):
        ckpt = tf.train.latest_checkpoint(name)
        if ckpt:
            self.model.load_weights(ckpt)
            print('Actor loaded ' + ckpt)

    def save(self, checkpoint_path):
        self.model.save_weights(checkpoint_path)

    def makestate(self, adj, wts_nn):
        reduced_nn = wts_nn.shape[0]
        # features = np.ones([reduced_nn, self.feature_size])
        features = np.multiply(np.ones([reduced_nn, self.feature_size]), wts_nn)
        support = simple_polynomials(adj, self.flags.max_degree)
        state = {"features": features, "support": support[1]}
        return state

    def memorize_crt(self, grad, loss, reward):
        self.memory_crt.append((grad.copy(), loss, reward))

    def memorize_act(self, grad, loss, reward):
        self.memory_act.append((grad.copy(), loss, reward))

    def predict(self, state):
        x_in = tf.convert_to_tensor(state["features"], dtype=tf.float64)
        coord, values, shape = state["support"]
        a_in = tf.sparse.SparseTensor(coord, values, shape)
        act_values = self.model([x_in, a_in])
        return act_values, np.argmax(act_values.numpy())

    def act(self, state, train):
        act_values, action = self.predict(state)
        return act_values, action  # returns action

    def predict_critic(self, z_out, state):
        coord, values, shape = state["support"]
        a_in = tf.sparse.SparseTensor(coord, values, shape)
        sch_pred = self.critic([z_out, a_in])
        return sch_pred

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return float('NaN'), float('NaN')
        self.reward_mem.clear()
        minibatch = random.sample(self.memory, batch_size)
        self.critic.trainable = True
        self.model.trainable = True
        for state, act_vals, solu, next_state, reward in minibatch:
            if next_state:
                act_vals_next, _ = self.predict(next_state)
                val_next = self.predict_critic(act_vals_next, next_state)
            else:
                val_next = np.zeros_like(act_vals)
            reward_t = tf.convert_to_tensor(reward, dtype=tf.float64) / 100.0
            reward_t = tf.reshape(reward_t, (-1, 1))
            val_vec = reward_t + self.gamma * val_next
            act_vals_t = tf.convert_to_tensor(act_vals, dtype=tf.float64)
            val_this = self.predict_critic(act_vals_t, state)
            with tf.GradientTape() as g:
                g.watch(self.critic.trainable_weights)
                regularization_loss = tf.reduce_sum(self.critic.losses)
                loss_value = tf.sqrt(self.mse(val_vec, val_this)) + regularization_loss
                gradients = g.gradient(loss_value, self.critic.trainable_weights)
                self.memorize_crt(gradients, loss_value.numpy(), reward)

            with tf.GradientTape() as g:
                g.watch(self.model.trainable_weights)
                act_val, _ = self.act(state, False)
                val_crt = self.predict_critic(act_val, state)
                regularization_loss = tf.reduce_sum(self.model.losses)
                obj_fn = -tf.reduce_mean(val_crt[:, 0])
                obj_fn = obj_fn + regularization_loss
                gradients = g.gradient(obj_fn, self.model.trainable_weights)
                self.memorize_act(gradients, obj_fn.numpy(), 0)

        loss_c = self.replay_crt(batch_size)
        loss_a = self.replay_act(batch_size*10)
        # self.memory.clear()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss_a, loss_c

    def replay_act(self, batch_size):
        if len(self.memory_act) < batch_size:
            return float('NaN')
        self.reward_mem.clear()
        minibatch = random.sample(self.memory_act, batch_size)
        losses = []
        for grad, loss, _ in minibatch:
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
            losses.append(loss)

        self.memory_act.clear()
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

    def utility(self, adj_0, wts_0, train=False):
        """
        GCN for per utility function
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals = self.act(state, train)

        gcn_wts = act_vals

        return gcn_wts, state

    def schedule(self, adj_0, wts_0, train=False):
        """
        GCN followed by LGS
        """
        adj = adj_0.copy()
        wts_nn = np.reshape(wts_0, (wts_0.shape[0], self.flags.feature_size))

        # GCN
        state = self.makestate(adj, wts_nn)
        act_vals_t, act = self.act(state, train)
        act_vals = act_vals_t.numpy()

        if self.flags.predict == 'mwis':
            gcn_wts = np.multiply(act_vals.flatten(), wts_nn.flatten())
        else:
            gcn_wts = act_vals.flatten()

        mwis, _ = local_greedy_search(adj, gcn_wts)
        solu = list(mwis)
        mwis_rt = mwis
        total_wt = np.sum(wts_nn[solu, 0])
        return mwis_rt, total_wt, state, act_vals

    def actor_train(self, adj_0, wts_0, train=False):
        adj = adj_0.copy()
        nn  = wts_0.shape[0]
        wts_nn = np.reshape(wts_0, (nn, FLAGS.feature_size))
        ones = np.ones_like(wts_nn)

        # GCN
        with tf.GradientTape() as g:
            g.watch(self.model.trainable_weights)
            state = self.makestate(adj, ones)
            act_val, act = self.act(state, train)
            act_val_norm = act_val
            # act_val_norm = 0.5 + (act_val - tf.reduce_mean(act_val))
            sch_pred = self.predict_critic(act_val_norm, state)

            if train:
                regularization_loss = tf.reduce_sum(self.model.losses)
                obj_fn = -tf.reduce_mean(sch_pred[:, 0])
                obj_fn = obj_fn + regularization_loss
                gradients = g.gradient(obj_fn, self.model.trainable_weights)
                self.memorize_act(gradients, obj_fn.numpy(), 0)
        return state, act_val

    def critic_train(self, adj_0, zs_0, state, n_samples=1, z_std=0.15):
        """
        GCN followed by LGS
        wts_0: topology weighted utility
        """
        adj = adj_0.copy()
        zs_nn = zs_0.numpy()
        nn = zs_nn.shape[0]

        # GCN
        with tf.GradientTape() as g:
            g.watch(self.critic.trainable_weights)
            sch_pred = self.predict_critic(zs_0, state)

            ind_vec = np.zeros_like(sch_pred.numpy())
            apu_avg = 0.0
            for i in range(n_samples):
                wts = np.random.uniform(0, 1, size=(nn, 1))
                # zs_i = zs_nn + np.random.normal(0, z_std, size=(nn, 1))
                zs_i = zs_nn + np.random.uniform(-0.5*z_std, 0.5*z_std, size=(nn, 1))
                if FLAGS.predict == 'mwis':
                    gcn_wts = np.multiply(zs_i.flatten(), wts.flatten())
                else:
                    gcn_wts = zs_i.flatten()

                _, total_ref = local_greedy_search(adj, wts)
                mwis, _ = local_greedy_search(adj, gcn_wts)
                solu = list(mwis)
                total_wt = np.sum(wts[solu, 0])
                ind_vec[solu, 0] += 1.0/float(n_samples)
                apu_avg += total_wt/(total_ref*float(n_samples)+1e-6)

            reward = apu_avg
            ind_vec[:, 1] = 1.0 - ind_vec[:, 0]
            y_target = tf.convert_to_tensor(ind_vec, dtype=tf.float64)
            regularization_loss = tf.reduce_sum(self.critic.losses)
            loss_value = tf.sqrt(self.mse(y_target, sch_pred)) + regularization_loss
            gradients = g.gradient(loss_value, self.critic.trainable_weights)
            self.memorize_crt(gradients, loss_value.numpy(), reward)
        return ind_vec, apu_avg


# use gpu 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

# Initialize session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
