#!/bin/bash

load=0.07
num_layer=1
graph='ba2'
name='TEST'
echo "load: ${load}, layer: ${num_layer}";

echo "python3 wireless_gcn_test_delay.py --wt_sel=qr --load_min=${load} --load_max=${load} --load_step=0.001 --feature_size=1 --epsilon=0.09 --epsilon_min=0.001 --diver_num=1 --datapath=./data/BA_Graph_Uniform_GEN21_test2 --test_datapath=./data/BA_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mis --hidden1=32 --num_layer=${num_layer} --instances=2 --training_set=${name} --opt=0 --gamma=0.9 --learning_rate=0.0001 --graph=${graph} > wireless/${graph}_${load}_l${num_layer}_${name}_qr_test.out ;"

python3 wireless_gcn_test_delay.py --wt_sel=qr --load_min=${load} --load_max=${load} --load_step=0.001 --feature_size=1 --epsilon=0.09 --epsilon_min=0.001 --diver_num=1 --datapath=./data/BA_Graph_Uniform_GEN21_test2 --test_datapath=./data/BA_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mis --hidden1=32 --num_layer=${num_layer} --instances=2 --training_set=${name} --opt=0 --gamma=0.9 --learning_rate=0.0001 --graph=${graph} > wireless/${graph}_${load}_l${num_layer}_${name}_qr_test.out ;

