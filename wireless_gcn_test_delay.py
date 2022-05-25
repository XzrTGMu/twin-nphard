#!/usr/bin/ python3
# -*- coding: utf-8 -*-
# python3
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
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
from graph_util import *

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

from agent_dqn_util import A2CAgent
from directory import find_model_folder

model_origin = find_model_folder(flags.FLAGS, 'exp')
flags1 = deepcopy(flags.FLAGS)
agent = A2CAgent(flags1, 64000)
try:
    agent.load(model_origin)
except:
    print("unable to load {}".format(model_origin))

n_instances = flags.FLAGS.instances


def emv(samples, pemv, n=3):
    assert samples.size == pemv.size
    k = float(2/(n+1))
    return samples * k + pemv * (1-k)


def channel_collision(adj, nflows, link_rates_ts, schedule_mv):
    """Return non-collision set of a schedule"""
    schedule = schedule_mv % nflows
    wts = np.zeros(shape=(nflows,), dtype=np.bool)
    if schedule.size > 0:
        wts[schedule] = 1
    non_collision = wts.copy()
    for s in schedule:
        _, nb_set = np.nonzero(adj[s])
        if np.sum(wts[nb_set]) > 0:
            non_collision[s] = 0
    capacity = np.zeros(shape=(nflows,))
    capacity[non_collision] = link_rates_ts[non_collision]
    return capacity


gtype = flags.FLAGS.graph
train = False
n_networks = 500
# n_instances = 10
timeslots = 64
lp = 5
algoname = 'DGCN-LGS'

algolist = ['Greedy', 'shadow', algoname]
if flags.FLAGS.opt == 0:
    algoname = 'DGCN-LGS'
elif flags.FLAGS.opt == 1:
    algoname = 'DGCN-LGS-it'
    algolist = [algoname]
elif flags.FLAGS.opt == 2 or flags.FLAGS.opt == 4:
    algoname = 'DGCN-RS'
    algolist = [algoname]
elif flags.FLAGS.opt == 3:
    algoname = 'CGCN-RS'
    algolist = [algoname]
else:
    sys.exit("Unsupported opt {}".format(flags.FLAGS.opt))

algoref = algolist[0]

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
                          'metric_vs_load_summary_{}-channel_utility-{}_opt-{}_graph-{}_load-{:.2f}_layer-{}_test.csv'
                          .format(n_ch, wt_sel, flags.FLAGS.opt, gtype, load_min, flags.FLAGS.num_layer)
                          )

res_list = []
res_df = pd.DataFrame(columns=['graph',
                               'seed',
                               'load',
                               'name',
                               'avg_queue_len',
                               '50p_queue_len',
                               '95p_queue_len',
                               '5p_queue_len',
                               'avg_utility',
                               'avg_degree'])
# if os.path.isfile(output_csv):
#     res_df = pd.read_csv(output_csv, index_col=0)

d_array = np.zeros((n_networks,), dtype=np.float)

if train:
    datapath = flags.FLAGS.datapath
    epochs = flags.FLAGS.epochs
else:
    datapath = flags.FLAGS.test_datapath
    epochs = 1

val_mat_names = sorted(os.listdir(datapath))

cnt = 0

# print("Average degree of all conflict graphs: {}".format(np.mean(d_array)))

np.random.seed(1)
if train:
    loss = 1.0
else:
    loss = np.nan

wts_sample_file = os.path.join(output_dir, 'samples.txt')

load_array = np.round(np.arange(load_min, load_max+load_step, load_step), 3)
# load = load_array[np.random.randint(0, len(load_array) - 1)]
load = load_min

buffer = deque(maxlen=20)
pemv = np.array([2.0])
pemv_best = np.array([1.05])
for i in range(100):
    np.random.seed(i+500)
    idx = i
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
    elif gtype == 'tree-line':
        try:
            graph_c = nx.random_powerlaw_tree(50, gamma=3.0, seed=i, tries=2000)
        except:
            graph_c = nx.random_powerlaw_tree(50, gamma=3.0, tries=1000)
        graph_i = nx.line_graph(graph_c)
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

    np.random.seed(seed)

    d_list = []
    for v in graph_i:
        d_list.append(graph_i.degree[v])
    avg_degree = np.nanmean(d_list)

    treeseed = int(1000 * time.time()) % 10000000
    np.random.seed(treeseed)

    arrival_rate = 0.5 * (sim_rate_lo + sim_rate_hi) * load

    interarrivals = np.random.exponential(1.0/arrival_rate, (nflows, int(2*timeslots*arrival_rate)))
    arrival_time = np.cumsum(interarrivals, axis=1)
    acc_pkts = np.zeros(shape=(nflows, timeslots))
    for t in range(0, timeslots):
        acc_pkts[:, t] = np.count_nonzero(arrival_time < t, axis=1)
    arrival_pkts = np.diff(acc_pkts, prepend=0)
    arrival_pkts = arrival_pkts.transpose()
    link_rates = np.random.normal(0.5 * (sim_rate_lo + sim_rate_hi), 0.25 * (sim_rate_hi - sim_rate_lo),
                                  size=[timeslots, nflows, n_ch])
    link_rates = link_rates.astype(int)
    link_rates[link_rates < sim_rate_lo] = sim_rate_lo
    link_rates[link_rates > sim_rate_hi] = sim_rate_hi

    to_print = []
    time_start = time.time()

    weight_samples = []
    queue_mtx_dict = {}
    dep_pkts_dict = {}
    util_mtx_dict = {}
    schedule_dict = {}
    wts_dict = {}
    queue_algo = np.zeros(shape=(lp, nflows))
    dep_pkts_algo = np.zeros(shape=(lp, nflows))
    queue_shadow = np.zeros(shape=(lp, nflows))
    dep_pkts_shadow = np.zeros(shape=(lp, nflows))
    wts_shadow = np.zeros(shape=(lp, nflows))
    for algo in algolist:
        queue_mtx_dict[algo] = np.zeros(shape=(timeslots, nflows))
        dep_pkts_dict[algo] = np.zeros(shape=(timeslots, nflows))
        util_mtx_dict[algo] = np.zeros(timeslots)
        schedule_dict[algo] = np.zeros(shape=(timeslots, nflows))
        util_mtx_dict[algo][0] = 1
        wts_dict[algo] = np.zeros(shape=(nflows, n_ch))

    state_buff = deque(maxlen=timeslots)
    mask_vec = np.arange(0, nflows)

    last_emb_vec = np.zeros(shape=(nflows*n_ch, ))
    last_sol_vec = np.zeros(shape=(nflows*n_ch, ))
    for t in range(1, timeslots):
        for algo in algolist:
            queue_mtx_dict[algo][t, :] = queue_mtx_dict[algo][t-1, :] + arrival_pkts[t, :]
            queue_mtx_algo = np.multiply(np.expand_dims(queue_mtx_dict[algo][t, :], axis=1), np.ones(shape=(nflows, n_ch)))
            if wt_sel == 'qr':
                wts0 = queue_mtx_algo * link_rates[t, :, :]
            elif wt_sel == 'q':
                wts0 = queue_mtx_algo
            elif wt_sel == 'qor':
                wts0 = queue_mtx_algo / link_rates[t, :, :]
            elif wt_sel == 'qrm':
                wts0 = np.minimum(queue_mtx_algo, link_rates[t, :, :])
            else:
                np.random.seed(i*1000+t)
                wts0 = np.random.uniform(0, 1, (nflows, n_ch))
            wts1 = np.reshape(wts0, nflows * n_ch, order='F')
            raw_wts = np.concatenate((queue_mtx_algo, link_rates[t, :, :]), axis=1)

            if algo == "Greedy":
                wts_dict[algo] = wts1
                mwis, total_wt = local_greedy_search(adj_gK, wts_dict[algo])
                mwis0, total_wt0 = greedy_search(adj_gK, wts_dict[algo])
                util_mtx_dict[algo][t] = total_wt/total_wt0
            elif algo == "Greedy-Th":
                wts_dict[algo] = wts1
                mwis, total_wt = dist_greedy_search(adj_gK, wts_dict[algo], 0.1)
                mwis0, total_wt0 = greedy_search(adj_gK, wts_dict[algo])
                util_mtx_dict[algo][t] = total_wt/total_wt0
            elif algo == 'Benchmark':
                wts_dict[algo] = wts1
                mwis, total_wt, _ = mlp_gurobi(adj_gK, wts_dict[algo])
                util_mtx_dict[algo][t] = 1.0
            elif algo == 'DGCN-LGS':
                wts_dict[algo] = wts1
                mwis0, total_wt0 = greedy_search(adj_gK, wts_dict[algo])
                act_vals, state = agent.utility(adj_gK, wts1, train=train)
                mwis, _ = local_greedy_search(adj_gK, act_vals)
                total_wt = np.sum(wts_dict[algo][list(mwis)])
                util_mtx_dict[algo][t] = total_wt / total_wt0
                state_buff.append((state, act_vals, list(mwis), t))
            elif algo == 'shadow':
                for ip in range(0, lp):
                    if ip == 0:
                        queue_shadow[0, :] = queue_mtx_dict[algoname][t-1, :] + arrival_pkts[t, :]
                    else:
                        if t + ip < timeslots:
                            queue_shadow[ip, :] = queue_shadow[ip-1, :] + arrival_pkts[t+ip, :]
                        else:
                            queue_shadow[ip, :] = queue_shadow[ip - 1, :]
                    queue_mtx_tmp = np.multiply(np.expand_dims(queue_shadow[ip, :], axis=1), np.ones(shape=(nflows, n_ch)))
                    if t + ip < timeslots:
                        wts_i = queue_mtx_tmp * link_rates[t+ip, :, :]
                        mwis, total_wt = local_greedy_search(adj_gK, wts_i)
                        schedule_mv = np.array(list(mwis))
                        link_rates_ts = np.reshape(link_rates[t+ip, :, :], nflows * n_ch, order='F')
                        capacity = channel_collision(adj_gK, nflows, link_rates_ts, schedule_mv)
                        dep_pkts_shadow[ip, :] = np.minimum(queue_shadow[ip, :], capacity)
                        queue_shadow[ip, :] = queue_shadow[ip, :] - dep_pkts_shadow[ip, :]
                    else:
                        dep_pkts_shadow[ip, :] = dep_pkts_shadow[ip-1, :]
                        queue_shadow[ip, :] = queue_shadow[ip-1, :]
                util_mtx_dict[algo][t] = 1
            elif algo == 'scheduler':
                wts_dict[algo] = wts1
                mwis0, total_wt0 = greedy_search(adj_gK, wts_dict[algo])
                mwis, actions, state = agent.scheduler(adj_gK, raw_wts, train=train)
                mwis, total_wt = local_greedy_search(adj_gK, wts_dict[algo]*actions)
                equal_wt = channel_collision(adj_gK, nflows, wts_dict[algo], np.array(list(mwis)))
                total_wt = np.sum(equal_wt)
                util_mtx_dict[algo][t] = total_wt / total_wt0
                state_buff.append((state, actions, mask_vec, t))
            else:
                sys.exit("Unsupported opt {}".format(flags.FLAGS.opt))

            schedule_mv = np.array(list(mwis))
            link_rates_ts = np.reshape(link_rates[t, :, :], nflows*n_ch, order='F')
            schedule_dict[algo][t, schedule_mv] = 1
            capacity = channel_collision(adj_gK, nflows, link_rates_ts, schedule_mv)
            if algo == 'shadow':
                dep_pkts_dict[algo][t, :] = np.mean(dep_pkts_shadow[:, :], axis=0)
                queue_mtx_dict[algo][t, :] = np.mean(queue_shadow[:, :], axis=0)
            else:
                dep_pkts_dict[algo][t, :] = np.minimum(queue_mtx_algo[:, 0], capacity)
                queue_mtx_dict[algo][t, :] = queue_mtx_dict[algo][t, :] - dep_pkts_dict[algo][t, :]

    avg_q_dict = {}
    med_q_dict = {}
    pct_q_dict = {}
    pct2_q_dict = {}
    avg_q_ts_dict = {}
    med_q_ts_dict = {}
    avg_q_links_dict = {}
    avg_dep_dict = {}
    energy_dict = {}
    for algo in algolist:
        avg_queue_length_ts = np.mean(queue_mtx_dict[algo], axis=1)
        med_queue_length_ts = np.median(queue_mtx_dict[algo], axis=1)
        pct_q_dict[algo] = np.percentile(queue_mtx_dict[algo], 95)
        pct2_q_dict[algo] = np.percentile(queue_mtx_dict[algo], 5)
        avg_queue_len_links = np.mean(queue_mtx_dict[algo], axis=0)
        avg_dep_dict[algo] = np.mean(dep_pkts_dict[algo])
        energy_dict[algo] = np.sum(schedule_dict[algo], axis=0)
        avg_q_links_dict[algo] = avg_queue_len_links
        avg_q_dict[algo] = np.mean(avg_queue_length_ts)
        med_q_dict[algo] = np.mean(med_queue_length_ts)
        avg_q_ts_dict[algo] = avg_queue_length_ts
        med_q_ts_dict[algo] = med_queue_length_ts
        std_flow_q = np.std(avg_queue_len_links)

        res_df = res_df.append({'graph': seed,
                                'seed': treeseed,
                                'load': load,
                                'name': algo,
                                'avg_queue_len': avg_q_dict[algo],
                                '50p_queue_len': med_q_dict[algo],
                                '95p_queue_len': pct_q_dict[algo],
                                '5p_queue_len': pct2_q_dict[algo],
                                'avg_utility': np.nanmean(util_mtx_dict[algo]),
                                'avg_degree': avg_degree
                                }, ignore_index=True)

    runtime = time.time() - time_start

    if wt_sel == 'random':
        buffer.append(np.mean(util_mtx_dict[algoname]))
    else:
        buffer.append(avg_q_dict[algoname]/avg_q_dict[algoref])
        pemv = emv(avg_q_dict[algoname]/avg_q_dict[algoref], pemv, 20)
    print("{}-{}: {}, load: {}, ".format(idx, i, netcfg, load),
        "q_med: {:.3f}, ".format(med_q_dict[algoname]/med_q_dict['Greedy']),
        "q_95: {:.3f}, ".format(pct_q_dict[algoname]/pct_q_dict['Greedy']),
        "q_avg: {:.3f}, ".format(avg_q_dict[algoname]/avg_q_dict['Greedy']),
        "d_avg: {:.3f}, ".format(avg_dep_dict[algoname]/avg_dep_dict['Greedy']),
        "u_gcn: {:.3f}, ".format(np.nanmean(util_mtx_dict[algoname])),
        "run: {:.3f}s, loss: a {:.5f}, c {:.5f}, ratio: {:.3f}, e: {:.4f} ".format(runtime, 1.0, 1.0, pemv[0], agent.epsilon),
        )
    i += 1

res_df.to_csv(output_csv, index=False)
# with open('./wireless/metric_vs_load_full.json', 'w') as fout:
#     json.dump(res_list, fout)

print("Done!")

