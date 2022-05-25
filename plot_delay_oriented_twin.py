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

dict_twin_l1 = {
    "Star30": "./wireless/star30_0.07_l1_GDPGsr_qr_test.out",
    "Star20": "./wireless/star20_0.07_l1_GDPGsr_qr_test.out",
    "Star10": "./wireless/star10_0.07_l1_GDPGsr_qr_test.out",
    "BA-m1": "./wireless/ba1_0.07_l1_GDPGsr_qr_test.out",
    "BA-m2": "./wireless/ba2_0.07_l1_GDPGsr_qr_test.out",
    "Tree": "./wireless/tree_0.07_l1_GDPGsr_qr_test.out",
    # "Tree": "./wireless/tree-line_0.07_l1_GDPGsr_qr_test.out",
    "ER": "./wireless/er_0.07_l1_GDPGsr_qr_test.out",
    # "poisson": "./wireless/poisson_0.07_l1_GDPGsr_test.out",
    "BA-mix": "./wireless/bamix_0.07_l1_GDPGsr_qr_test.out"
}

dict_df = {}
results = {}
for item in ['q_med', 'q_95', 'q_avg','u_gcn']:
    df_tmp = pd.DataFrame({item:[],'graph':[],'PADR':[],'baseline':[],'train':[]})
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
        df_item['train'] = 'Lookahead RL [Zhao22]'
        df_tmp = df_tmp.append(df_item)

    for key in dict_twin_l1:
        fullpath = dict_twin_l1[key]
        file = fullpath.split('/')[-1]
        # fullpath = os.path.join(result_dir, file)
        if file.endswith(".out"):
            print(fullpath)
        else:
            continue

        df = pd.read_csv(fullpath, header=None)
        df.columns = header_soj
        for col in header_soj[4:]:
            df[col] = pd.to_numeric(df[col].str.replace('s','').str.split(' ').str[-1], errors='coerce')
        results[fullpath] = df
        df_item = pd.DataFrame([])
        df_item[item] = df[item]
        df_item['graph'] = "{}\n{}".format(key, padr[key])
        df_item['PADR'] = padr[key]
        df_item['baseline'] = 'queue'
        df_item['train'] = 'GDPG-Twin'
        df_tmp = df_tmp.append(df_item)
    dict_df[item] = df_tmp


fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 4))

toplot = ['q_med', 'q_avg', 'u_gcn']
# toplot = ['q_95', 'q_med', 'q_avg', 'sj_95p', 'sj_med', 'sj_avg']
xlabels = []
padr = padr.sort_values()
for index, val in padr.items():
    if index in dict_out_l1.keys():
        xlabels.append("{}\n{}".format(index, val))

plotitems = ['Median', 'Mean', '$95^{th}$']
ylims = [[], [], []]
for i in range(2):
    item = toplot[i]
    df_tmp = dict_df[item]
    ax = axs[i%3]
    if i <3:
        x_offset = -0.2
    else:
        x_offset = 0.2
    # boxplot = df_tmp.boxplot(column=[item], by='PADR', rot=0, ax=ax,
    #                          color=dict(boxes='b', whiskers='k', medians='r', caps='k'),
    #                          showmeans=True, return_type='dict',
    #                          widths=0.3, positions=np.arange(1, 9) + x_offset)
    flierprops = dict(marker='o', markerfacecolor='None', markersize=3,
                      linestyle='none')
    ax = sns.boxplot(x="PADR", y=item, hue="train",
                     data=df_tmp, flierprops=flierprops,
                     color=dict(boxes='b', whiskers='k', medians='r', caps='k'),
                     ax=ax, palette="tab20", showmeans=True)

    ax.set_title('')
    ax.set_ylabel(r'{} Backlog'.format(plotitems[i%3]), fontsize=12)
    # boxplot.set_xlabel('Topology')
    ax.set_xlabel('')
    ax.set_xticks(list(range(0, 8)))
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_ylim(0.48, 1.22)
    ax.grid(True)
    if i == 0:
        ax.get_legend().remove()
    # ax.legend(['RL-lookahead', "GDPG-Twin"])
    # df_means = df_tmp.groupby('PADR').mean()
    # x = 0
    # for index, row in df_means.iterrows():
    #     y = row[item]
    #     text = '{:.3f}'.format(y)
    #     ax.annotate(text, xy=(x + x_offset, y),
    #                 xytext=(0, -15), textcoords="offset pixels",
    #                 color='g',
    #                 fontsize=12,
    #                 horizontalalignment="center",
    #                 verticalalignment="top")
    #     x += 1
    print(item)

fig.suptitle('')
fig.subplots_adjust(wspace=0)
plt.tight_layout()
fig.subplots_adjust(hspace=0)
plt.subplots_adjust(left=0.09, right=0.995, top=0.995, bottom=0.11)
plt.savefig("./output/delay_oriented_qr_by_topology_twin.png",
            dpi = 300,  # facecolor='w', edgecolor='w',
            orientation = 'portrait',
            format = 'png')
plt.savefig("./output/delay_oriented_qr_by_topology_twin.pdf",
            dpi = 300,  # facecolor='w', edgecolor='w',
            orientation = 'portrait',
            format = 'pdf')
print("done")
plt.close(fig)

