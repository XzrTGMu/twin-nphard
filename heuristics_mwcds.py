import networkx as nx
import dwave_networkx as dnx
import igraph as ig
import pulp as plp
from pulp import GLPK
import numpy as np
import pandas as pd
import scipy.sparse as sp
import copy
import time
print(nx.__version__)


def zero_rows(M, rows):
    diag = sp.eye(M.shape[0]).tolil()
    for r in rows:
        diag[r, r] = 0
    return diag.dot(M)


def zero_columns(M, columns):
    diag = sp.eye(M.shape[1]).tolil()
    for c in columns:
        diag[c, c] = 0
    return M.dot(diag)


# From @wim's post
def nunique(a):
    df = pd.DataFrame(a.T)
    return df.nunique().to_numpy(dtype=np.float)


def steiner_terminal_2hop(adj, wts):
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    # verts = np.array(range(wts_0.size))
    terminals = set()
    adj_2 = adj_0.dot(adj_0)
    verts = wts_0.argsort()
    verts = verts.tolist()
    while len(verts) > 0:
        i = verts.pop(0)
        _, nb1_set = np.nonzero(adj_0[i])
        _, nb2_set = np.nonzero(adj_2[i])
        terminals.add(i)
        verts = list(set(verts) - set(nb1_set))
        verts = list(set(verts) - set(nb2_set))
    return terminals


def mwds_greedy_mis(adj, wts):
    '''
    Ref: Ant Colony Optimization Applied to Minimum Weighted Dominating Set Problem
    Color: White: Uncovered, Black: Dominating, Gray: Covered
    Return MWDS set and the total weights of MWDS
    :param adj: adjacency matrix (sparse)
    :param wts: weights of vertices
    :return: mwds, total_wt
    '''
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    verts = np.array(range(wts_0.size))
    mwds = set()
    gray_set = set()
    white_set = set(verts)
    wts_1 = wts_0.copy()
    while len(white_set) > 0:
        covers = adj_0.dot(wts_1)
        # covers = adj_0.dot(np.diag(np.eye(wts_0.size)))
        weights = np.divide(wts_0, 1 + covers)
        # weights = wts_0 - covers
        # weights = np.divide(1 + covers, wts_0)
        weights[list(gray_set)] = np.Inf
        weights[list(mwds)] = np.Inf
        # weights[covers == 0] = -1
        i = np.argmin(weights)
        # if covers[i] == 0:
        #     continue
        _, nb_set = np.nonzero(adj_0[i])
        mwds.add(i)
        nb_set = set(nb_set).intersection(white_set)
        gray_set = gray_set.union(nb_set)
        nb_set.add(i)
        white_set = white_set - nb_set - mwds
        rm_set = list(nb_set)
        wts_1[rm_set] = 0
        # adj_0 = zero_rows(adj_0, rm_set)
        # adj_0 = zero_columns(adj_0, rm_set)
    total_ws = np.sum(wts[list(mwds)])
    return mwds, gray_set, total_ws


def mwds_greedy(adj, wts):
    '''
    Ref: Ant Colony Optimization Applied to Minimum Weighted Dominating Set Problem
    Color: White: Uncovered, Black: Dominating, Gray: Covered
    Return MWDS set and the total weights of MWDS
    :param adj: adjacency matrix (sparse)
    :param wts: weights of vertices
    :return: mwds, total_wt
    '''
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    verts = np.array(range(wts_0.size))
    mwds = set()
    gray_set = set()
    white_set = set(verts)
    wts_1 = wts_0.copy() + 1e-6
    degrees = adj_0.sum(axis=1)
    degrees = np.asarray(degrees).flatten()
    while len(white_set) > 0:
        covers = adj_0.dot(wts_1)
        # covers = adj_0.dot(np.diag(np.eye(wts_0.size)))
        weights = np.divide(wts_0, 1 + covers)
        # weights = wts_0 - covers
        weights[list(mwds)] = np.Inf
        weights[np.logical_and(covers == 0, degrees > 0)] = np.Inf
        i = np.argmin(weights)
        _, nb_set = np.nonzero(adj_0[i])
        mwds.add(i)
        nb_set = set(nb_set).intersection(white_set)
        gray_set = gray_set.union(nb_set)
        nb_set.add(i)
        white_set = white_set - nb_set - mwds
        rm_set = list(nb_set)
        wts_1[rm_set] = 0
    total_ws = np.sum(wts[list(mwds)])
    return mwds, gray_set, total_ws


def node_weight_steiner_set(adj, wts, mis):
    # Shortest Path Heuristic
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    nodes = np.array(range(wts_0.size))
    mwcds = copy.deepcopy(mis)
    mwcds_list = list(mwcds)
    gray_set = set(nodes) - mwcds
    nodes_forest = np.full_like(nodes, np.nan, dtype=np.double)
    nodes_forest[mwcds_list] = mwcds_list
    nodes_forest += 1.0 # forest id initialized as node id
    cover_forest = np.full_like(nodes, np.nan, dtype=np.double)
    # Merge any connected trees
    for i in mwcds_list:
        _, nb_set = np.nonzero(adj_0[i])
        nb_tid = nodes_forest[nb_set]
        nb_tids = nb_tid[~np.isnan(nb_tid)]
        cover_forest[i] = len(set(nb_set).intersection(gray_set))
        # nb_trees = nb_set[~np.isnan(nb_tid)]
        if len(nb_tids) > 0:
            fid_max = np.nanmax(nb_tid)
            nodes_forest[i] = fid_max
            for tid in nb_tids:
                nodes_forest[nodes_forest == tid] = fid_max
    # Creates a quotient graph
    tids = np.unique(nodes_forest[mwcds_list])
    trees = [set(np.argwhere(nodes_forest==tid).flatten()) for tid in tids]
    forest = []
    for tree in trees:
        tree_dict = {'head': max(tree), 'member': tree, 'cover': np.sum(cover_forest[list(tree)])}
        forest.append(tree_dict)

    wts_st = copy.deepcopy(wts_0)
    wts_st[mwcds_list] = 0

    def edge_weight(s, d, attr):
        return wts_st[s]+wts_st[d]
    g = nx.from_scipy_sparse_matrix(adj_0)
    start = max(forest, key=lambda x: x['cover'])
    forest.remove(start)
    while len(forest) > 0:
        dists = []
        paths = []
        for tree in forest:
            src = start['head']
            dst = tree['head']
            path = nx.shortest_path(g, src, dst, weight=edge_weight)
            dists.append(np.sum(wts_st[path]))
            paths.append(path)
        pid = np.argmin(dists)
        path = paths[pid]
        tree = forest[pid]
        wts_st[path] = 0
        newtree = start['member'].union(tree['member']).union(set(path))
        newhead = max(newtree)
        start['head'] = newhead
        start['member'] = newtree
        forest.remove(tree)
    mwcds = start['member']
    total_ws = np.sum(wts_0[list(mwcds)])
    return mwcds, total_ws


def nwst_sph(adj, wts, mis):
    # Shortest Path Heuristic, no sorting
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    nodes = np.array(range(wts_0.size))
    mwcds = copy.deepcopy(mis)
    mwcds_list = list(mwcds)
    gray_set = set(nodes) - mwcds
    nodes_forest = np.full_like(nodes, np.nan, dtype=np.double)
    nodes_forest[mwcds_list] = mwcds_list
    nodes_forest += 1.0 # forest id initialized as node id
    cover_forest = np.full_like(nodes, np.nan, dtype=np.double)
    # Merge any connected trees
    for i in mwcds_list:
        _, nb_set = np.nonzero(adj_0[i])
        nb_tid = nodes_forest[nb_set]
        nb_tids = nb_tid[~np.isnan(nb_tid)]
        cover_forest[i] = len(set(nb_set).intersection(gray_set))
        # nb_trees = nb_set[~np.isnan(nb_tid)]
        if len(nb_tids) > 0:
            fid_max = np.nanmax(nb_tid)
            nodes_forest[i] = fid_max
            for tid in nb_tids:
                nodes_forest[nodes_forest == tid] = fid_max
    # Creates a quotient graph
    tids = np.unique(nodes_forest[mwcds_list])
    trees = [set(np.argwhere(nodes_forest==tid).flatten()) for tid in tids]
    forest = []
    for tree in trees:
        tree_dict = {'head': max(tree), 'member': tree, 'cover': np.sum(cover_forest[list(tree)])}
        forest.append(tree_dict)

    wts_st = copy.deepcopy(wts_0)
    wts_st[mwcds_list] = 0

    def edge_weight(s, d, attr):
        return wts_st[s]+wts_st[d]
    g = nx.from_scipy_sparse_matrix(adj_0)
    # start = max(forest, key=lambda x: x['cover'])
    start = forest[0]
    forest.remove(start)
    while len(forest) > 0:
        dists = []
        paths = []
        for tree in forest:
            src = start['head']
            dst = tree['head']
            path = nx.shortest_path(g, src, dst, weight=edge_weight)
            dists.append(np.sum(wts_st[path]))
            paths.append(path)
        pid = np.argmin(dists)
        path = paths[pid]
        tree = forest[pid]
        wts_st[path] = 0
        newtree = start['member'].union(tree['member']).union(set(path))
        newhead = max(newtree)
        start['head'] = newhead
        start['member'] = newtree
        forest.remove(tree)
    mwcds = start['member']
    total_ws = np.sum(wts_0[list(mwcds)])
    return mwcds, total_ws


def steiner_tree_mst(adj, wts, terminals):
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    nodes = np.array(range(wts_0.size))
    graph = nx.from_scipy_sparse_matrix(adj_0)
    for u in graph:
        graph.nodes[u]['weight'] = wts_0[u]
    mwcds = copy.deepcopy(terminals)
    mwcds_list = list(mwcds)
    wts_st = copy.deepcopy(wts_0)
    wts_st[mwcds_list] = 0

    def edge_weight(s, d, attr):
        return wts_st[s]+wts_st[d]

    total_ws = 0.0
    sg = nx.algorithms.approximation.steinertree.steiner_tree(graph, terminals, weight=edge_weight)
    st = list(sg.nodes)
    mwcds = set(st)
    total_ws = np.sum(wts_0[st])
    return mwcds, total_ws


def greedy_mwcds(adj, wts):
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    # step 1: find MWDS, which is an independent set
    mwds, gray_set, _ = mwds_greedy_mis(adj_0, wts_0)
    # step 2: find node weighted steiner tree, with MWDS as the set of terminals
    nodes = np.array(range(wts_0.size))
    mwcds = copy.deepcopy(mwds)
    mwds_list = list(mwds)
    nodes_forest = np.full_like(nodes, np.nan, dtype=np.double)
    nodes_forest[mwds_list] = mwds_list
    nodes_forest += 1
    gray_list = np.array(list(gray_set))
    gray_wts = wts_0[gray_list]
    idx = np.argsort(gray_wts)
    gray_list = gray_list[idx]
    degrees = adj_0.sum(axis=1)
    degrees = np.asarray(degrees).flatten()
    for i in gray_list:
        if degrees[i] <= 1:
            continue
        _, nb_set = np.nonzero(adj_0[i])
        nb_fid = nodes_forest[nb_set]
        nb_fid_set = set(nb_fid[~np.isnan(nb_fid)])
        if len(nb_fid_set) > 1:
            mwcds.add(i)
            fid_max = np.nanmax(nb_fid)
            nodes_forest[i] = fid_max
            for fid in nb_fid_set:
                nodes_forest[nodes_forest==fid] = fid_max

    total_ws = np.sum(wts_0[list(mwcds)])
    return mwcds, mwds, total_ws


def greedy_mwcds2(adj, wts):
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    # step 1: find MWDS, which is an independent set
    # mwds, gray_set, _ = mwds_greedy_mis(adj_0, wts_0)
    mwds, gray_set, _ = mwds_greedy(adj_0, wts_0)
    # step 2: find node weighted steiner tree, with MWDS as the set of terminals
    mwcds, total_ws = node_weight_steiner_set(adj_0, wts_0, mwds)
    return mwcds, mwds, total_ws


def mwcds_vvv(adj, wts):
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    # step 1: find MWDS, which is an independent set
    graph = nx.from_scipy_sparse_matrix(adj_0)
    for u in graph:
        graph.nodes[u]['weight'] = wts_0[u]
    mwds = nx.algorithms.approximation.min_weighted_dominating_set(graph, 'weight')
    # step 2: find node weighted steiner tree, with MWDS as the set of terminals
    mwcds, total_ws = node_weight_steiner_set(adj_0, wts_0, mwds)
    return mwcds, mwds, total_ws


def mwds_vvv(adj, wts):
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    # step 1: find MWDS, which is an independent set
    graph = nx.from_scipy_sparse_matrix(adj_0)
    for u in graph:
        graph.nodes[u]['weight'] = wts_0[u]
    mwds = nx.algorithms.approximation.min_weighted_dominating_set(graph, 'weight')
    total_ws = np.sum(wts_0[list(mwds)])
    gray_set = set(range(wts_0.size)) - mwds
    return mwds, gray_set, total_ws


def dist_greedy_mwds(adj, wts):
    '''
    Ref: Ant Colony Optimization Applied to Minimum Weighted Dominating Set Problem (weight heuristic)
    Color: White: Uncovered, Black: Dominating, Gray: Covered
    Return MWDS set and the total weights of MWDS
    :param adj: adjacency matrix (sparse)
    :param wts: weights of vertices
    :return: mwds, total_wt
    '''
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    verts = np.array(range(wts_0.size))
    mwds = set()
    gray_set = set()
    white_set = set(verts)
    while len(white_set) > 0:
        # weights = np.divide(1 + adj_0.dot(np.diag(np.eye(wts_0.size))), wts_0)
        weights = np.divide(wts_0, 1 + adj_0.dot(wts_0))
        weights[list(gray_set)] = 0
        weights[list(mwds)] = 0
        for i in white_set:
            proc = False
            _, nb_set = np.nonzero(adj_0[i])
            if len(nb_set) > 0:
                nb_min = np.amin(weights[nb_set])
                nb_min_set = nb_set[weights[nb_set] == nb_min]
                if weights[i] < nb_min or (weights[i] == nb_min and i < np.amax(nb_min_set)):
                    proc = True
            else:
                proc = True
            if proc:
                mwds.add(i)
                nb_set = set(nb_set).intersection(white_set)
                gray_set = gray_set.union(nb_set)
                nb_set.add(i)
        rm_set = gray_set.union(mwds)
        white_set = white_set - rm_set
        adj_0 = zero_rows(adj_0, list(rm_set))
        adj_0 = zero_columns(adj_0, list(rm_set))
    total_ws = np.sum(wts_0[list(mwds)])
    return mwds, gray_set, total_ws


def dist_node_weight_steiner_set(adj, wts, mis):
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    nodes = np.array(range(wts_0.size))
    mwcds = copy.deepcopy(mis)
    mwcds_list = list(mwcds)
    gray_set = set(nodes) - mwcds
    nodes_forest = np.full_like(nodes, np.nan, dtype=np.double)
    nodes_forest[mwcds_list] = mwcds_list
    nodes_forest += 1.0 # forest id initialized as node id
    cover_forest = np.full_like(nodes, np.nan, dtype=np.double)
    # Merge any connected trees
    for i in mwcds_list:
        _, nb_set = np.nonzero(adj_0[i])
        nb_tid = nodes_forest[nb_set]
        nb_tids = nb_tid[~np.isnan(nb_tid)]
        cover_forest[i] = len(set(nb_set).intersection(gray_set))
        # nb_trees = nb_set[~np.isnan(nb_tid)]
        if len(nb_tids) > 0:
            fid_max = np.nanmax(nb_tid)
            nodes_forest[i] = fid_max
            for tid in nb_tids:
                nodes_forest[nodes_forest == tid] = fid_max
    # Creates a quotient graph
    tids = np.unique(nodes_forest[mwcds_list])
    trees = [set(np.argwhere(nodes_forest==tid).flatten()) for tid in tids]
    forest = []
    for tree in trees:
        tree_dict = {'head': max(tree), 'member': tree}
        forest.append(tree_dict)

    wts_st = copy.deepcopy(wts_0)
    wts_st[mwcds_list] = 0

    def edge_weight(s, d, attr):
        return wts_st[s]+wts_st[d]
    g = nx.from_scipy_sparse_matrix(adj_0)
    while len(forest) > 1:
        dst_paths = []
        dst_trees = []
        for start in forest:
            dists = []
            paths = []
            for tree in forest:
                if tree['head'] == start['head']:
                    continue
                src = start['head']
                dst = tree['head']
                path = nx.shortest_path(g, src, dst, weight=edge_weight)
                dists.append(np.sum(wts_st[path]))
                paths.append(path)
            pid = np.argmin(dists)
            path = paths[pid]
            # tree = forest[pid]
            dst_trees.append(pid)
            dst_paths.append(path)
        for tid in range(len(forest)):
            src_tree = forest[tid]
            dst_tree = forest[dst_trees[tid]]
            path = dst_paths[tid]
            wts_st[path] = 0
            new_tree = src_tree['member'].union(dst_tree['member']).union(set(path))
            new_head = max(new_tree)
            new_item = {'head': new_head, 'member': new_tree}
            forest[tid] = new_item
            forest[dst_trees[tid]] = new_item
        # forest = list(set(forest))
        exclude_list = []
        del_list = []
        for tid in range(len(forest)):
            src_tree = forest[tid]
            exclude_list.append(tid)
            for ttid in range(len(forest)):
                if ttid in exclude_list:
                    continue
                else:
                    dst_tree = forest[ttid]
                if len(src_tree['member'].intersection(dst_tree['member'])) > 0:
                    new_tree = src_tree['member'].union(dst_tree['member'])
                    new_head = max(new_tree)
                    new_item = {'head': new_head, 'member': new_tree}
                    src_tree = new_item
                    exclude_list.append(ttid)
                    del_list.append(ttid)
            forest[tid] = src_tree
            tid += 1
        forest = [forest[idx] for idx in range(len(forest)) if idx not in del_list]

    mwcds = forest[0]['member']
    total_ws = np.sum(wts_0[list(mwcds)])
    return mwcds, total_ws


def dist_greedy_mwcds(adj, wts):
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    # step 1: find MWDS, which is an independent set
    mwds, gray_set, _ = dist_greedy_mwds(adj_0, wts_0)
    mwcds, total_ws = node_weight_steiner_set(adj_0, wts_0, mwds)
    return mwcds, mwds, total_ws


def dist_greedy_mwcds2(adj, wts):
    '''
    Return MWCDS set and the total weights of MWCDS
    :param adj: adjacency matrix (sparse)
    :param wts: weights of vertices
    :param epislon: 0<epislon<1, to determin alpha and beta
    :return: mwis, total_wt
    '''
    adj_0 = adj.copy()
    wts_0 = np.array(wts).flatten()
    # step 1: find MWDS, which is an independent set
    mwds, gray_set, _ = dist_greedy_mwds(adj_0, wts_0)
    # step 2: find node weighted steiner tree, with MWDS as the set of terminals
    nodes = np.array(range(wts_0.size))
    mwcds = copy.deepcopy(mwds)
    mwcds_list = list(mwcds)
    leafs = set()
    nodes_forest = np.full_like(nodes, np.nan, dtype=np.double)
    nodes_forest[mwcds_list] = mwcds_list
    nodes_forest += 1.0
    # Merge any connected trees
    for i in mwcds_list:
        _, nb_set = np.nonzero(adj_0[i])
        nb_tid = nodes_forest[nb_set]
        nb_tids = nb_tid[~np.isnan(nb_tid)]
        # nb_trees = nb_set[~np.isnan(nb_tid)]
        if len(nb_tids) > 0:
            fid_max = np.nanmax(nb_tid)
            nodes_forest[i] = fid_max
            for tid in nb_tids:
                nodes_forest[nodes_forest == tid] = fid_max

    # adj_2 = adj_0.dot(adj_0)
    while np.unique(nodes_forest[mwcds_list]).size > 1:
        fid_mtx = adj_0.dot(np.diag(nodes_forest))
        fid_mtx[fid_mtx == 0] = np.nan
        fcover = nunique(fid_mtx)
        # fid_mtx2 = adj_2.dot(np.diag(nodes_forest))
        # fid_mtx2[fid_mtx2 == 0] = np.nan
        # fcover2 = nunique(fid_mtx2)
        weights = np.divide(wts_0, 1 + fcover)
        weights[list(mwcds)] = np.Inf
        weights[list(leafs)] = np.Inf
        gray_set = gray_set - mwcds - leafs
        for i in gray_set:
            # if fcover[i] <= 1:
            #     leafs.add(i)
            #     continue
            _, nb_set = np.nonzero(adj_0[i])
            nb_min = np.amin(weights[nb_set])
            nb_min_set = nb_set[weights[nb_set] == nb_min]
            self_cover = np.unique(fid_mtx[i])
            self_cover = self_cover[~np.isnan(self_cover)].tolist()
            nb_weights = np.zeros_like(nb_set, dtype=np.float)
            pair_cov_map = {}
            for j in range(len(nb_set)):
                nb_v = nb_set[j]
                _, nb_pair_set = np.nonzero(adj_0[nb_v])
                nb_fids = nodes_forest[nb_pair_set]
                nb_cover = np.unique(nb_fids[~np.isnan(nb_fids)]).tolist()
                pair_cover = set(self_cover).union(set(nb_cover))
                p_cover = float(len(pair_cover))
                wts_pair = (wts_0[i] + wts_0[nb_v])/(1+p_cover)
                pair_cov_map[nb_v] = pair_cover
                if p_cover > 1:
                    nb_weights[j] = wts_pair
                else:
                    nb_weights[j] = np.Inf
            nbp_min = np.amin(nb_weights)
            nbvp_min = min(nb_min, nbp_min)

            if (weights[i] < nbvp_min and fcover[i] > 1) or (weights[i] == nbvp_min and i < np.amax(nb_min_set)):
                nb_fid = nodes_forest[nb_set]
                nb_fid_set = set(nb_fid[~np.isnan(nb_fid)])
                if len(nb_fid_set) > 1:
                    mwcds.add(i)
                    fid_max = np.nanmax(nb_fid)
                    nodes_forest[i] = fid_max
                    for fid in nb_fid_set:
                        nodes_forest[nodes_forest == fid] = fid_max
            elif nbp_min <= nbvp_min:
                # a pair is the smallest
                nbp_set = nb_set[nb_weights == nbp_min]
                nbp = np.amin(nbp_set)
                nb_fid_set = pair_cov_map[nbp]
                if len(nb_fid_set) > 1:
                    mwcds.add(i)
                    mwcds.add(nbp)
                    fid_max = np.nanmax(list(nb_fid_set))
                    nodes_forest[i] = fid_max
                    nodes_forest[nbp] = fid_max
                    for fid in nb_fid_set:
                        nodes_forest[nodes_forest == fid] = fid_max
            elif fcover[i] <= 1:
                leafs.add(i)

        mwcds_list = list(mwcds)

    total_ws = np.sum(wts_0[list(mwcds)])
    return mwcds, mwds, total_ws


def test_heuristic():
    # Create a random graph
    t = time.time()
    seed = np.random.randint(1, 10000)
    # seed = 1844
    # seed = 5251
    # graph = nx.gaussian_random_partition_graph(200, 10, 10, 0.25, 0.1, seed=seed)
    graph = nx.generators.random_graphs.barabasi_albert_graph(200, 10)
    wts = np.random.uniform(0.1, 1.1, (200,))
    N = len(graph.nodes)
    for u in graph:
        graph.nodes[u]['weight'] = wts[u]  # np.square(np.random.randn())
        # graph.nodes[u]['id'] = u
    print("Time to create graph: {:.3f} s, graph seed: {}, connected: {}\n".format(time.time()-t, seed, nx.is_connected(graph)))
    # Run Neighborhood Removal
    adj = nx.adjacency_matrix(graph, nodelist=list(range(N)))
    weights = np.array(wts)
    vertices = np.array(range(N))

    t = time.time()
    # mwds = nx.algorithms.approximation.min_weighted_dominating_set(graph, 'weight')
    # mwcds, total_wt = node_weight_steiner_set(adj, weights, set(mwds))
    mwcds, mwds, total_wt = mwcds_vvv(adj, weights)
    subgraph = graph.subgraph(mwcds)
    print("min_weighted_dominating_set: {:.3f} s".format(time.time()-t))
    print("Original Graph: {} nodes, {} edges.".format(graph.number_of_nodes(), graph.number_of_edges()))
    print("Total Weights: {:.3f}, DS size: {}, CDS size: {}\n{}".format(total_wt, len(mwds), len(mwcds), mwcds))
    print("Validation: dominate {}, connected {}\n".format(nx.is_dominating_set(graph, mwds), nx.is_connected(subgraph)))

    t = time.time()
    mwds = nx.algorithms.maximal_independent_set(graph)
    mwds = set(mwds)
    mwcds, total_wt = node_weight_steiner_set(adj, weights, mwds)
    # mwcds, total_wt = steiner_tree_mst(adj, weights, mwds)
    subgraph = graph.subgraph(mwcds)
    print("maximal_independent_set + SPH: {:.3f} s".format(time.time()-t))
    print("Original Graph: {} nodes, {} edges.".format(graph.number_of_nodes(), graph.number_of_edges()))
    print("Total Weights: {:.3f}, DS size: {}, CDS size: {}\n{}".format(total_wt, len(mwds), len(mwcds), mwcds))
    print("Validation: dominate {}, connected {}\n".format(nx.is_dominating_set(graph, mwds), nx.is_connected(subgraph)))

    t = time.time()
    mwds = nx.algorithms.maximal_independent_set(graph)
    mwds = set(mwds)
    # mwcds, total_wt = node_weight_steiner_set(adj, weights, mwds)
    mwcds, total_wt = steiner_tree_mst(adj, weights, mwds)
    subgraph = graph.subgraph(mwcds)
    print("maximal_independent_set + MST: {:.3f} s".format(time.time()-t))
    print("Original Graph: {} nodes, {} edges.".format(graph.number_of_nodes(), graph.number_of_edges()))
    print("Total Weights: {:.3f}, DS size: {}, CDS size: {}\n{}".format(total_wt, len(mwds), len(mwcds), mwcds))
    print("Validation: dominate {}, connected {}\n".format(nx.is_dominating_set(graph, mwds), nx.is_connected(subgraph)))

    t = time.time()
    mwcds1, mwds1, total_wt1 = greedy_mwcds2(adj, weights)
    subgraph = graph.subgraph(mwcds1)
    print("Time of greedy search 1: {:.3f} s".format(time.time()-t))
    print("Original Graph: {} nodes, {} edges.".format(graph.number_of_nodes(), graph.number_of_edges()))
    print("Total Weights: {:.3f}, DS size: {}, CDS size: {}\n{}".format(total_wt1, len(mwds1), len(mwcds1), mwcds1))
    print("Validation: dominate {}, connected {}\n".format(nx.is_dominating_set(graph, mwds1), nx.is_connected(subgraph)))

    # t = time.time()
    # mwcds0, mwds0, total_wt = greedy_mwcds(adj, weights)
    # subgraph = graph.subgraph(mwcds0)
    # print("Time of greedy search 0: {:.3f} s".format(time.time()-t))
    # print("Original Graph: {} nodes, {} edges.".format(graph.number_of_nodes(), graph.number_of_edges()))
    # print("Total Weights: {:.3f}, DS size: {}, CDS size: {}\n{}".format(total_wt, len(mwds0), len(mwcds0), mwcds0))
    # print("Validation: dominate {}, connected {}\n".format(nx.is_dominating_set(graph, mwds0), nx.is_connected(subgraph)))

    t = time.time()
    mwcds, mwds, total_wt = dist_greedy_mwcds(adj, weights)
    subgraph = graph.subgraph(mwcds)
    print("Time of distributed greedy search (1-hop): {:.3f} s".format(time.time()-t))
    print("Original Graph: {} nodes, {} edges.".format(graph.number_of_nodes(), graph.number_of_edges()))
    print("Total Weights: {:.3f}, DS size: {}, CDS size: {}\n{}".format(total_wt, len(mwds), len(mwcds), mwcds))
    print("Validation: dominate {}, connected {}\n".format(nx.is_dominating_set(graph, mwds), nx.is_connected(subgraph)))

    t = time.time()
    mwcds, mwds, total_wt = dist_greedy_mwcds2(adj, weights)
    subgraph = graph.subgraph(mwcds)
    print("Time of distributed greedy search (2-hop): {:.3f} s".format(time.time()-t))
    print("Original Graph: {} nodes, {} edges.".format(graph.number_of_nodes(), graph.number_of_edges()))
    print("Total Weights: {:.3f}, DS size: {}, CDS size: {}\n{}".format(total_wt, len(mwds), len(mwcds), mwcds))
    print("Validation: dominate {}, connected {}\n".format(nx.is_dominating_set(graph, mwds), nx.is_connected(subgraph)))


if __name__ == "__main__":
    test_heuristic()
