#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091


source shell_experiments/run.sh
# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

# DATA=("cifar10_alpha0.8" )
DATA=("emnist20")

n_rounds=200

sampling_rates=( "0.5")

# run_avg

# mus=("0.05" "0.1" "0.2" "0.5" "1")
# run_prox
# run_l2gd
# run_pfedme

# alphas=("0.5"  "0.25")
# run_apfl

pre_rounds_list=("50")

trans_list=("0.75")
fuzzy_m_momentums=("0.8")
# fuzzy_m_schedulers=("constant")
measurements=("graddot")

fuzzy_m_schedulers=("constant")
fuzzy_m_list=("1.7")
run_fuzzy

# fuzzy_m_schedulers=("cosine_annealing")

# fuzzy_m_list=("1.6")
# min_m_list=("1.5")
# run_fuzzy

# m_list=("1.75")
# min_m_list=("1.6" "1.65")
# run_fuzzy

# fuzzy_m_list=("1.5")
# min_m_list=("1.4")
# run_fuzzy

# fuzzy_m_list=("1.8")
# min_m_list=("1.7")
# run_fuzzy

# fuzzy_m_list=("2")
# min_m_list=("1.8")
# run_fuzzy

# fuzzy_m_schedulers=("constant")
# fuzzy_m_list=("1.6" "1.75" "1.5")

# run_avgem

# DATA=("femnist")

# run_em

# run_l2gd