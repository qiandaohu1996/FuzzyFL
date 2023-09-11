#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh

# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

# DATA=("cifar10_alpha0.8" )
DATA=("cifar100" )

# DATA=("cifar10")
n_rounds=200

sampling_rates=( "1")

 
mus=("0.1" "0.5"  "1")
# run_prox
run_l2gd
run_pfedme
mus=( "0.1"  "1")
run_prox

# pre_rounds_list=("50")
# fuzzy_m_list=( "1.6")
# min_m_list=("1.5")
# trans_list=("0.75")
# fuzzy_m_momentums=("0.5" )
# fuzzy_m_schedulers=("cosine_annealing")
# # fuzzy_m_schedulers=("constant")
# measurements=("loss" "euclid")

# run_fuzzy

# fuzzy_m_list=( "1.75")
# min_m_list=("1.6") 
# run_fuzzy

# fuzzy_m_list=( "1.5")
# min_m_list=("1.4") 

# run_fuzzy

# run_avgem

# DATA=("femnist")

# run_em



alphas=("0.5")

run_apfl
DATA=("cifar10_alpha0.8" "cifar10_pathologic_cl3" )

mus=("0.1" "1")

run_prox
run_pfedme
# run_l2gd