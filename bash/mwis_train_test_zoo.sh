#!/bin/bash

setval='MWISZOO'
layers=3
echo "MWIS ZOO"
echo "Training starts"
python3 mwis_gcn_train_zoo.py --training_set=${setval} --epsilon=1.0 --epsilon_min=0.002 --gamma=0.99 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_mixN_mixp_train0 --test_datapath=./data/ER_Graph_Uniform_GEN21_test1 --max_degree=1 --predict=mwis --learning_rate=0.0001 --learning_decay=1.0 --hidden1=32 --num_layer=3 --epochs=25 --ntrain=1 > ./output/mwis_training_trace_zoo.out ;
echo "Testing starts"
python3 mwis_test_complexity.py --training_set=${setval} --epsilon=1 --epsilon_min=0.002 --feature_size=1 --diver_num=1 --datapath=./data/ER_Graph_Uniform_GEN21_test2 --max_degree=1 --predict=mwis --learning_rate=0.00001 --hidden1=32 --num_layer=3 --opt=0 ;
echo "MWIS ZOO done"
