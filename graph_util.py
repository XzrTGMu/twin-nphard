import networkx as nx
# from networkx.algorithms.approximation import independent_set
import numpy as np
import scipy.sparse as sp
from scipy.io import savemat
from scipy.spatial import distance_matrix
import dwave_networkx as dnx
import os
from itertools import chain, combinations
from heuristics import greedy_search


def power_set(iterable):
    """power_set([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def random_node_weights(graph, dist, max_wts=1.0):
    """
    Generate random node weights for input graph according to dist
    """
    if dist.lower() == 'uniform':
        for u in graph:
            graph.nodes[u]['weight'] = np.random.uniform(0, max_wts)
    elif dist.lower() == 'normal_l1':
        for u in graph:
            graph.nodes[u]['weight'] = np.abs(np.random.randn())
    elif dist.lower() == 'normal_l2':
        for u in graph:
            graph.nodes[u]['weight'] = np.square(np.random.randn())
    return graph


def weighted_random_graph(N, p, dist, max_wts=1.0):
    """
    create a random ER graph
    """
    graph = nx.generators.random_graphs.fast_gnp_random_graph(N, p)
    graph = random_node_weights(graph, dist, max_wts)
    return graph


def weighted_poisson_graph(area, density, radius=1.0, dist='uniform', max_wts=1.0):
    """
    Create a Poisson point process 2D graph
    """
    N = np.random.poisson(lam=area*density)
    lenth_a = np.sqrt(area)
    xys = np.random.uniform(0, lenth_a, (N, 2))
    d_mtx = distance_matrix(xys, xys)
    adj_mtx = np.zeros([N, N], dtype=int)
    adj_mtx[d_mtx <= radius] = 1
    np.fill_diagonal(adj_mtx, 0)
    graph = nx.from_numpy_matrix(adj_mtx)
    graph = random_node_weights(graph, dist, max_wts)
    return graph


def weighted_barabasi_albert_graph(N, p, dist, max_wts=1.0):
    graph = nx.generators.random_graphs.barabasi_albert_graph(N, int(np.round(N*p)))
    graph = random_node_weights(graph, dist, max_wts)
    return graph


# maximum weighted independent set
def mwis_heuristic_1(graph):
    adj_0 = nx.adj_matrix(graph).todense()
    wts = np.array([graph.nodes[u]['weight'] for u in graph.nodes])
    a = -wts
    vec_is = -np.ones(adj_0.shape[0])
    while np.any(vec_is == -1):
        rem_vector = vec_is == -1
        adj = adj_0.copy()
        adj = adj[rem_vector, :]
        adj = adj[:, rem_vector]

        u = np.argmin(a[rem_vector].dot(adj != 0)/a[rem_vector])
        n_is = -np.ones(adj.shape[0])
        n_is[u] = 1
        neighbors = np.argwhere(adj[u, :] != 0)
        if neighbors.shape[0]:
            n_is[neighbors] = 0
        vec_is[rem_vector] = n_is
    #print(IS)
    mwis1 = np.nonzero(vec_is > 0)[0]
    val = np.sum(wts[mwis1])
    # print("Total Weight: {}".format(val))
    # print(mwis1)
    # print(dnx.is_independent_set(graph, mwis1))
    return mwis1, val


def mwis_heuristic_2(graph):
    mis_set = []
    mwis = []
    maxval = 0
    for u in graph:
        mis = nx.maximal_independent_set(graph, [u])
        # print(mis)
        mis_set.append(mis)
        val = 0
        for u in mis:
            val += graph.nodes[u]['weight']
        if val > maxval:
            maxval = val
            mwis = mis
    # mis_set
    # print(maxval)
    # print(mwis)
    # print(dnx.is_independent_set(graph, mwis))
    return mwis, maxval


def mwis_heuristic_greedy(graph):
    adj = nx.adjacency_matrix(graph)
    weights = np.array([graph.nodes[u]['weight'] for u in graph])
    mwis, total_wt = greedy_search(adj, weights)
    return mwis, total_wt


def mis_check(adj, mis):
    graph = nx.from_scipy_sparse_matrix(adj)
    result = dnx.is_independent_set(graph, mis)
    return result


def mwis_bruteforce(graph):
    adj = nx.adjacency_matrix(graph)
    weights = np.array([graph.nodes[u]['weight'] for u in graph])
    vertices = list(range(len(weights)))
    p_sets = power_set(vertices)
    mwis = []
    maxweights = 0.0
    cnt = 0
    for p_set in p_sets:
        cnt += 1
        if len(p_set) == 0:
            continue
        l_set = list(p_set)
        if not dnx.is_independent_set(graph, l_set):
            continue
        utility = np.sum(weights[l_set])
        if utility > maxweights:
            mwis = l_set
            maxweights = utility
    return mwis, maxweights


def poisson_graphs_from_dict(gdict):
    adj_c = gdict['adj_c']
    adj_i = gdict['adj_i']
    # d_mtx = gdict['d_mtx']
    xys = gdict['xys']

    # generate connectivity graph
    np.fill_diagonal(adj_c, 0)
    graph_c = nx.from_numpy_matrix(adj_c)
    for u in graph_c:
        graph_c.nodes[u]['xy'] = xys[u, :]

    # generate conflict graph
    graph_cf = nx.from_numpy_matrix(adj_i)

    return graph_c, graph_cf


def connection_graph_poisson(adj_c, xys):
    """
    Generate connection graph with xy cordinates of nodes
    """
    # generate connectivity graph
    np.fill_diagonal(adj_c, 0)
    graph_c = nx.from_numpy_matrix(adj_c)
    for u in graph_c:
        graph_c.nodes[u]['xy'] = xys[u, :]
    return graph_c


def multichannel_conflict_simulate(adj_i, k=3, p=0.8):
    """
    Generate multiple conflict graphs from a base conflict graph
    input: adj_i, base conflict graph
    input: k, number of instances
    input: p, probability of overlapping edges
    output: a list of conflict graphs with the same vertex set
    """
    # generate conflict graph
    if not sp.issparse(adj_i):
        adj_i = sp.csr_matrix(adj_i)
    graphs_cf = []
    for c in range(k):
        graph_cf = nx.from_scipy_sparse_matrix(adj_i)
        for u in graph_cf:
            for v in graph_cf:
                if u <= v:
                    continue
                if adj_i[u, v]:
                    if np.random.rand() > p:
                        graph_cf.remove_edge(u, v)
        graphs_cf.append(graph_cf)
    return graphs_cf


def multichannel_conflict_graph(graphs):
    """
    Generate multi-channel conflict graph from conflict graphs on channels
    input: graphs, a list of conflict graphs with the same vertex set
    output: adj_list, a list of adjacency matrices of input graphs
    output: adj_gK, multi-channel conflict graph as an adjacency matrix
    """
    # for multiGCN inputs
    adj_list = []
    # for centralized scheduling
    graph_K = nx.Graph()
    nk = len(graphs)
    no_nodes = []
    for k in range(nk):
        g = graphs[k]
        iu = 0
        no_nodes.append(g.number_of_nodes())
        for u in g:
            j = k * no_nodes[-1] + iu
            graph_K.add_node(j, weight=1.0, name='({},{})'.format(iu, k))
            iu += 1
    assert(len(set(no_nodes)) == 1)
    nn = no_nodes[0]
    # add interface constraint for single radio
    for n in range(nn):
        for k1 in range(nk):
            v1 = k1*nn + n
            for k2 in range(nk):
                if k1 >= k2:
                    continue
                v2 = k2*nn + n
                graph_K.add_edge(v1, v2)
    for k in range(nk):
        g = graphs[k]
        adj = nx.adjacency_matrix(g)
        adj_list.append(adj)
        for e in g.edges:
            v1, v2 = e
            graph_K.add_edge(k*nn+v1, k*nn+v2)

    adj_gK = nx.adjacency_matrix(graph_K)
    return adj_list, adj_gK


def degree_centralization(graph):
    """https://www.sciencedirect.com/topics/computer-science/degree-centrality"""
    V = graph.number_of_nodes()
    H = (V-1) * (V-2)
    # degs = np.zeros([V, ])
    # for n, deg in graph.degree():
    #     degs[n] = deg
    degs = []
    for n, deg in graph.degree():
        degs.append(deg)
    degs = np.array(degs)
    deg_max = np.amax(degs)
    deg_cent = np.sum(deg_max - degs) / H
    return deg_cent, degs

