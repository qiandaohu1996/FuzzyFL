#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh

n_rounds=200

pre_rounds_list=("1")
learner_rates=( "0.1")

sampling_rates=( "0.5")
# sampling_rates=( "0.5")
n_clusters=3
top=3

commands=()
# seed=2222

DATA=(
    # "emnist_c5_topk" 
    # "emnist_c5_alpha0.2" 
    # "emnist_c5_alpha0.4" 
    # "emnist_c5_alpha0.6" 
    # "emnist_c5_alpha0.8" 
    # "emnist_c10_alpha0.2"
    # "emnist_c10_alpha0.3"
    # "emnist_c10_alpha0.4"
    # "emnist_c10_alpha0.5"
    # "emnist_c10_alpha0.6"
    # "emnist_n200_c10_alpha0.3" 
    # "emnist_n200_c10_alpha0.4"
    # "emnist_c10_topk" 
    # "emnist" 
    # "emnist_alpha0.2" 
    # "emnist_alpha0.6"  
    # "emnist_alpha0.8" 
    # "emnist_pathologic_cl5" 
    # "emnist_pathologic_cl10" 
    # "emnist_pathologic_cl20" 
    "cifar100"
    )


# print_programs

# multi_run

 

fuzzy_m_momentums=("0.8")
trans_list=("0.75")
n_clusters=3
top=3
# n_clusters=5
# top=5


# fuzzy_m_list=( "1.6" )
# measurements=( "euclid")
# fuzzy_m_schedulers=( "constant")
# 
# fuzzy_m_list=("1.5" "1.7" "1.8" "2")
# fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2" "2.2")
# fuzzy_m_list=("1.5" "2" "2.2")
# fuzzy_m_list=("1.6")
# fuzzy_m_schedulers=( "cosine_annealing")
# measurements=("euclid")
# get_fuzzy_cmd 
# fuzzy_m_schedulers=("constant")
# fuzzy_m_list=("1.5")
# get_fuzzy_cmd 
fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2" "2.2")

# trans_list=("0.9" "0.75")
measurements=("loss")
fuzzy_m_schedulers=("cosine_annealing" )
get_fuzzy_cmd 


max_concurrent_processes=3

print_programs
multi_run
# commands=()
