#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run2.sh

# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

DATA=("shakespeare_s0.3" )

# DATA=("cifar10")
n_rounds=200

sampling_rates=("1")

pre_rounds_list=("50")

trans_list=("0.75")
fuzzy_m_momentums=("0.8")
fuzzy_m_schedulers=("cosine_annealing")
# fuzzy_m_schedulers=("constant")
measurements=("loss" "euclid")




fuzzy_m_list=("1.75")
min_m_list=("1.6") 
run_fuzzy


fuzzy_m_list=("1.6")
min_m_list=("1.5")
run_fuzzy

fuzzy_m_list=("1.8")
min_m_list=("1.7") 

run_fuzzy
