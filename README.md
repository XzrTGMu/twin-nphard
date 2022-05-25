# Graph-based Deterministic Policy Gradient for Repetitive Combinatorial Optimization Problems

This repository is part of the paper with the given title, that has been submitted to NeurIPS 2022.
This README contains instructions for replicating the 5 numerical experiments demonstrated in the paper.
The code is based on Python3 + Tensorflow 2.

## Abstract
We propose an actor-critic framework for graph-based machine learning pipelines with non-differentiable blocks, and apply it to repetitive combinatorial optimization problems (COPs) under hard constraints.
Repetitive COP refers to problems to be solved repeatedly on graphs of the same or slowly changing topology but rapidly changing node or edge weights.
Compared to one-shot COPs, repetitive COPs often rely on fast heuristics to solve one instance of the problem before the next one arrives, at the cost of a relatively large optimality gap.
Through numerical experiments on several discrete optimization problems, we show that our approach can learn reusable node or edge representations to reduce the optimality gap of fast heuristics for independent repetitive COPs,
and can even enable the system to account for dependencies between consecutive COPs to optimize long-term objectives.

## Setup

The following intructions assume that Python3.9 is your default Python3.
Other versions of Python3 may also work. 

`pip3 install -r requirements.txt`

Install any missing packages while running the code or notebook.

## Directory

```bash
├── bash # bash commands
├── data # training and testing datasets
├── doc # documents
├── gcn # GCN module used by baseline
├── model # Trained models
├── output # Raw outputs of COPs
├── wireless # Raw outputs of delay-oriented scheduling 
├── plot_test_results.ipynb # Scripts of figure plotting
├── plot_training.ipynb # Plotting training curves
├── LICENSE
├── README.md
└── requirements.txt
```

## 1. Maximum Weighted Independent Set (MWIS)

Trained models

GDPG-Twin
`./model/result_GCNTwinAlt_deep_ld1_c32_l3_cheb1_diver1_mwis_dqn` 

GDPG-Twin (critic twin network)
`./model/result_GCNTwinAlt_deep_ld1_c32_l3_cheb1_diver1_mwis_critic` 

ZOO
`./model/result_GCNZoo4_deep_ld1_c32_l3_cheb1_diver1_mwis_dqn`

Ad hoc RL
`./model/result_IS4SAT_deep_ld1_c32_l3_cheb1_diver1_mwis_dqn`


Generating training and testing datasets

```bash
cd .. # To the upper directory of the root of this project
git clone https://github.com/zhongyuanzhao/distgcn.git
cd distgcn
bash bash/run_data_generation.sh
```

copy generated datasets from `../distgcn/data/` to `./data/` of this project.

See <https://github.com/zhongyuanzhao/distgcn> for more information.

### 1.1 GDPG-Twin

```bash
setval='MWISTwin'
layers=3
echo "MWIS GDPG-Twin"
echo "Training starts"
python3 mwis_gcn_train_twin.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=3 --epochs=25 --ntrain=1 ;
echo "Testing starts"
python3 mwis_test_complexity.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=3 --opt=0 ;
echo "MWIS Twin done"
```

### 1.2 Zeroth order optimization

```bash
setval='MWISZOO'
layers=3
echo "MWIS ZOO"
echo "Training starts"
python3 mwis_gcn_train_zoo.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=3 --epochs=25 --ntrain=1 ;
echo "Testing starts"
python3 mwis_test_complexity.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=3 --opt=0 ;
echo "MWIS ZOO done"
```

### 1.3 Ad hoc Reinforcement Learning

```bash
cd ../distgcn/
layers=3
setval='MWISAdhoc'
echo "MWIS ad hoc RL"
echo "Training starts"
python3 mwis_dqn_origin.py --training_set=${setval}  --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=5
python3 mwis_dqn_origin.py --training_set=${setval}  --epsilon=0.2 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=5
python3 mwis_dqn_origin.py --training_set=${setval}  --epsilon=0.1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.000001 --hidden1=32 --num_layer=${layers} --epochs=5
python3 mwis_dqn_origin.py --training_set=${setval}  --epsilon=0.05 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.0000001 --hidden1=32 --num_layer=${layers} --epochs=10

echo "Testing starts"

testfolder20="ER_Graph_Uniform_GEN21_test2";
testfolder22="BA_Graph_Uniform_GEN21_test2";

dist='Uniform'
python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder20} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=10
mv ./output/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn_${testfolder20}.csv

python3 mwis_dqn_test.py --training_set=${setval} --epsilon=.0002 --feature_size=1 --diver_num=1 --datapath=./data/${testfolder22} --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=${layers} --epochs=10
mv ./output/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn.csv ./output/result_${setval}_deep_ld1_c32_l${layers}_cheb1_diver1_mwis_dqn_${testfolder22}.csv

echo "MWIS ad hoc RL done"
```

See <https://github.com/zhongyuanzhao/distgcn> for more information.


## 2. Minimum Weighted Dominating Set (MWDS)

Trained models

Actor

`./model/result_GCNDSGDYER_deep_ld1_c32_l5_cheb1_diver1_mpy_dpg_policy`

Critic

`./model/result_GCNDSGDYER_deep_ld1_c32_l5_cheb1_diver1_mpy_critic`



Make sure lines 46 - 48 in `mwds_gcn_call_twin.py` are as follows

```python
# heuristic_func = mwds_greedy_mis
heuristic_func = mwds_greedy
# heuristic_func = mwds_vvv
```

### 2.1 GDPG-Twin

```bash
setval="GCNDSGDY"
echo "MWDS training starts"
python3 mwds_gcn_train_twin.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='er'
echo "MWDS test starts"
python3 mwds_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/WS_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='ws'
python3 mwds_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/BA_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='ba'
python3 mwds_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='er'
python3 mwds_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/GRP_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='grp'
```


## 3. Node Weighted Steiner Tree (NWST)

Trained models

Actor

`./model/result_GCNSteinerGRP_deep_ld1_c32_l5_cheb1_diver1_mpy_dpg_policy`

Critic

`./model/result_GCNSteinerGRP_deep_ld1_c32_l5_cheb1_diver1_mpy_critic`


### 3.1 GDPG-Twin
```bash
setval="GCNSTGRP"
echo "NWST training starts"
python3 steiner_gcn_train_twin.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='grp'
echo "NWST test starts"
python3 steiner_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/WS_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='ws'
python3 steiner_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/BA_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='ba'
python3 steiner_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='er'
python3 steiner_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ --test_datapath=./data/GRP_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='grp'
```

## 4. Minimum Weighted Connected Dominating Set (MWCDS)

Trained models

Actor

`./model/result_GCNCDS_deep_ld1_c32_l5_cheb1_diver2_mpy_dpg_policy`

Critic

`./model/result_GCNCDS_deep_ld1_c32_l5_cheb1_diver2_mpy_critic`


Make sure lines 46 - 48 in `mwcds_gcn_call_twin.py` are as follows

```python
heuristic_func = dist_greedy_mwcds
# heuristic_func = mwcds_vvv
# heuristic_func = greedy_mwcds2
```


### 4.1 GDPG-Twin

```bash
setval="GCNCDSMIS"
echo "MWCDS training starts"
python3 mwcds_gcn_train_twin.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=2 --datapath=./data/ --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='grp'
echo "MWCDS test starts"
python3 mwcds_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=2 --datapath=./data/ --test_datapath=./data/WS_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='ws'
python3 mwcds_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=2 --datapath=./data/ --test_datapath=./data/BA_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='ba'
python3 mwcds_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=2 --datapath=./data/ --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='er'
python3 mwcds_gcn_test.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=2 --datapath=./data/ --test_datapath=./data/GRP_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mpy --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=5 --epochs=20 --ntrain=1 --gtype='grp'
```


## 5. Delay-oriented distributed link scheduling

Trained models

GDPG-Twin

Actor `./model/result_GDPGsr_deep_ld1_c32_l1_cheb1_diver1_mis_gdpg`

Critic `./model/result_GDPGsr_deep_ld1_c32_l1_cheb1_diver1_mis_gdpg_crt`

Lookahead RL (Ref)

`./model/result_STARBA2_deep_ld1_c32_l1_cheb1_diver1_mis_exp`

Quick test for Figure 5

```bash
bash bash/wireless_gcn_delay_test_twin.sh
bash bash/wireless_gcn_delay_test.sh
```

### 5.1 GDPG-Twin

Train

```bash
python3 wireless_gcn_train_delay_twin.py --wt_sel=qr --load_min=0.05 --load_max=0.05 --load_step=0.002 --feature_size=1 --epsilon=0.09 --epsilon_min=0.001 --diver_num=1 --datapath=./data/BA_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/BA_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mis --hidden1=32 --num_layer=1 --instances=2 --training_set=DelayTwin --opt=0 --gamma=0.95 --learning_rate=0.001 --graph=ba2
```

Test

Change line 3 in `bash/wireless_gcn_delay_test_twin.sh` to `setval='DelayTwin'`, then run `bash bash/wireless_gcn_delay_test_twin.sh`.

Note that when `--graph=ba2`, the datasets specified by `datapath` and `test_datapath` are not used, the training graphs are generated on the fly.


### 5.2 Lookahead Reinforcement Learning

Train

```bash
python3 wireless_gcn_train_delay.py --wt_sel=qr --load_min=0.05 --load_max=0.05 --load_step=0.002 --feature_size=1 --epsilon=0.09 --epsilon_min=0.001 --diver_num=1 --datapath=./data/BA_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/BA_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mis --hidden1=32 --num_layer=1 --instances=2 --training_set=DelayLHRL --opt=0 --gamma=0.9 --learning_rate=0.0001 --graph=ba2
```

see <https://github.com/zhongyuanzhao/gcn-dql> for more information.

Change line 3 in `bash/wireless_gcn_delay_test.sh` to `setval='DelayLHRL'`, then run `bash bash/wireless_gcn_delay_test.sh`.



## 6. Core References

### 6.1 Major Codebases

1. <https://github.com/zhongyuanzhao/distgcn>
2. Delay-oriented distributed link scheduling <https://github.com/zhongyuanzhao/gcn-dql>
3. Graph convolutional neural network: 1) spektral <https://graphneural.network/>,  2) `./gcn/` <https://github.com/tkipf/gcn>

### 6.2 Algorithms

1. Maximum Weighted Independent Set (MWIS): local greedy solver [[Joo 2012](https://ieeexplore.ieee.org/document/5714691)], code [[Zhao 2021](https://github.com/zhongyuanzhao/distgcn/blob/main/heuristics.py)]
2. Minimum Weighted Dominating Set (MWDS) 
3. Node Weighted Steiner Tree (NWST) 
4. Minimum Weighted Connected Dominating Set (MWCDS) 
5. Zeroth order optimization (ZOO) 

