import os
import sys
import scipy.io as sio
import scipy.sparse as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns
import re
numeric_const_pattern = r"""
     [-+]? 
     (?:
         (?: \d* \. \d+ )
         |
         (?: \d+ \.? )
     )
     (?: [Ee] [+-]? \d+ ) ?
     """
rx = re.compile(numeric_const_pattern, re.VERBOSE)

header = ['Config', 'n', 'f', 't', 'load','q_med', 'q_95', 'q_avg', 'd_avg', 'u_gcn','run','loss a', 'loss c', 'ratio',
          'e']

header_soj = ['Config', 'n', 'f', 't', 'load','q_med', 'q_95', 'q_avg', 'd_avg', 'u_gcn', 'sj_avg', 'sj_med', 'sj_95p',
              'run', 'ratio', 'e']


padr = pd.read_csv('./wireless/graph_centrality.csv', index_col='graph')
padr = padr["PADR"]


dict_out_l1 = {
    "Star30": "./wireless/star30_0.07_l1_GCNBP2_qr_test.out",
    "Star20": "./wireless/star20_0.07_l1_GCNBP2_qr_test.out",
    "Star10": "./wireless/star10_0.07_l1_GCNBP2_qr_test.out",
    "BA-m1": "./wireless/ba1_0.07_l1_GCNBP2_qr_test.out",
    "BA-m2": "./wireless/ba2_0.07_l1_GCNBP2_qr_test.out",
    "Tree": "./wireless/tree_0.07_l1_GCNBP2_qr_test.out",
    # "Tree": "./wireless/tree-line_0.07_l1_GCNBP2_qr_test.out",
    "ER": "./wireless/er_0.07_l1_GCNBP2_qr_test.out",
    # "poisson": "./wireless/poisson_0.07_l1_GCNBP2_test.out",
    "BA-mix": "./wireless/bamix_0.07_l1_GCNBP2_qr_test.out"
}


dict_df = {}
results = {}
for item in ['q_med', 'q_95', 'q_avg','u_gcn']:
    df_tmp = pd.DataFrame({item:[],'graph':[],'PADR':[],'baseline':[]})
    for key in dict_out_l1:
        fullpath = dict_out_l1[key]
        file = fullpath.split('/')[-1]
        if file.endswith(".out"):
            print(fullpath)
        else:
            continue

        df = pd.read_csv(fullpath, header=None)
        df.columns = header
        for col in header[4:]:
            df[col] = pd.to_numeric(df[col].str.replace('s','').str.split(' ').str[-1], errors='coerce')
        results[fullpath] = df
        df_item = pd.DataFrame([])
        df_item[item] = df[item]
        df_item['graph'] = "{}\n{}".format(key, padr[key])
        df_item['PADR'] = padr[key]
        df_item['baseline'] = 'queue'
        df_tmp = df_tmp.append(df_item)
    dict_df[item] = df_tmp

fig, axs = plt.subplots(3, 1, sharex='all', figsize=(6, 6))

toplot = ['q_95', 'q_med', 'q_avg', 'u_gcn']
xlabels = []
padr = padr.sort_values()
for index, val in padr.items():
    if index in dict_out_l1.keys():
        xlabels.append("{}\n{}".format(index, val))

plotitems = ['$95^{th}$', 'Median', 'Mean']
ylims = [[], [], []]
for i in range(3):
    item = toplot[i]
    df_tmp = dict_df[item]
    ax = axs[i%3]
    if i <3:
        x_offset = -0.2
    else:
        x_offset = 0.2
    boxplot = df_tmp.boxplot(column=[item], by='PADR', rot=0, ax=ax,
                             color=dict(boxes='b', whiskers='k', medians='r', caps='k'),
                             showmeans=True, return_type='dict',
                             widths=0.3, positions=np.arange(1, 9) + x_offset)
    ax.set_title('')
    ax.set_ylabel(r'AR ({})'.format(plotitems[i%3]), fontsize=12)
    ax.set_xlabel('')
    ax.set_xticks(list(range(1, 9)))
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylim(0.48, 1.22)
    df_means = df_tmp.groupby('PADR').mean()
    x = 1
    for index, row in df_means.iterrows():
        y = row[item]
        text = '{:.3f}'.format(y)
        ax.annotate(text, xy=(x + x_offset, y),
                    xytext=(0, -15), textcoords="offset pixels",
                    color='g',
                    fontsize=12,
                    horizontalalignment="center",
                    verticalalignment="top")
        x += 1
    print(item)

fig.suptitle('')
fig.subplots_adjust(wspace=0)
plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig("./output/delay_oriented_qr_by_topology.png",
            dpi = 300,  # facecolor='w', edgecolor='w',
            orientation = 'portrait',
            format = 'png')
plt.savefig("./output/delay_oriented_qr_by_topology.pdf",
            dpi = 300,  # facecolor='w', edgecolor='w',
            orientation = 'portrait',
            format = 'pdf')
print("done")
plt.close(fig)

df_load = pd.DataFrame({'seed':[],'load':[],'graph':[],'q_med':[], 'q_95':[], 'q_avg':[],'u_gcn':[]})
num_layer = 1

for graph in ['star20', 'star10', 'ba1', 'ba2', 'tree', 'er']:
    for load in np.arange(0.01, 0.09, 0.01):
        file = 'metric_vs_load_summary_1-channel_utility-qr_opt-0_graph-{}_load-{:.2f}_layer-{}_test.csv'.format(graph, load, num_layer)
        fullpath = os.path.join('./wireless/', file)
        df = pd.read_csv(fullpath, index_col=False)
        df_greedy = df.loc[df['name']=='Greedy']
        df_greedy.set_index('graph', inplace=True)
        df_gcn = df.loc[df['name']=='DGCN-LGS']
        df_gcn.set_index('graph', inplace=True)
        df_out = df_greedy[['seed', 'load']]
        df_out['q_med'] = df_gcn['50p_queue_len']/df_greedy['50p_queue_len']
        df_out['q_95'] = df_gcn['95p_queue_len']/df_greedy['95p_queue_len']
        df_out['q_avg'] = df_gcn['avg_queue_len']/df_greedy['avg_queue_len']
        df_out['u_gcn'] = df_gcn['avg_utility']/df_greedy['avg_utility']
        df_out.reset_index(drop=True, inplace=True)
        df_out['graph'] = graph
        df_load = df_load.append(df_out, ignore_index=True)

for item in ['q_avg', 'q_med']:
    fig = plt.figure(figsize=(5, 3))
    ax = sns.lineplot(data=df_load, x="load", y=item, hue="graph", style='graph', markers=True) # ['.','+','*','x','o','^']
    ax.set_xlabel("x: traffic load", fontsize=12)
    ax.set_ylabel("Approx. Ratio to LGS", fontsize=12)
    ax.grid('both')
    ax.legend(['Star20', 'Star10', 'BA-m1', 'BA-m2', 'Tree', 'ER'])
    plt.tight_layout(pad=0.1)
    fig.savefig("./output/delay_oriented_{}_qr_by_load.png".format(item),
                dpi=300, #facecolor='w', edgecolor='w',
                orientation='portrait',
                format='png')
    fig.savefig("./output/delay_oriented_{}_qr_by_load.pdf".format(item),
                dpi=300, #facecolor='w', edgecolor='w',
                orientation='portrait',
                format='pdf')
    plt.close(fig)
print("Done")