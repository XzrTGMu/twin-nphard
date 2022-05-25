#!/bin/bash


for graph in 'star30' 'star20' 'star10' 'ba1' 'ba2' 'tree' 'er'; do
	python3 wireless_degree_centrality.py --wt_sel=qrm --load_min=0.07 --load_max=0.07 --load_step=0.02 --feature_size=2 --epsilon=0.09 --epsilon_min=0.001 --diver_num=1 --datapath=./data/BA_Graph_Uniform_GEN21_test2 --test_datapath=./data/wireless_test --max_degree=1 --predict=mis --hidden1=32 --num_layer=3 --instances=2 --training_set=STARF2 --opt=0 --gamma=0.9 --learning_rate=0.0001 --graph=${graph} &
done

# graph='poisson'
python3 wireless_degree_centrality.py --wt_sel=qrm --load_min=0.07 --load_max=0.07 --load_step=0.02 --feature_size=2 --epsilon=0.09 --epsilon_min=0.001 --diver_num=1 --datapath=./data/BA_Graph_Uniform_GEN21_test2 --test_datapath=./data/wireless_test --max_degree=1 --predict=mis --hidden1=32 --num_layer=3 --instances=2 --training_set=STARF2 --opt=0 --gamma=0.9 --learning_rate=0.0001 --graph=poisson &

# graph='ba150'
python3 wireless_degree_centrality.py --wt_sel=qrm --load_min=0.07 --load_max=0.07 --load_step=0.02 --feature_size=2 --epsilon=0.09 --epsilon_min=0.001 --diver_num=1 --datapath=./data/BA_Graph_Uniform_GEN21_test2 --test_datapath=./data/BA_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mis --hidden1=32 --num_layer=3 --instances=2 --training_set=STARF2 --opt=0 --gamma=0.9 --learning_rate=0.0001 --graph=ba150 &
