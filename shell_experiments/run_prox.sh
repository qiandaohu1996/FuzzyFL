#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh
# DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("emnist")
# learner_rates=("0.05")
learner_rates=("0.1")
# mus=("0.02" ) 
commands=()
mus=("0.05" "0.1" "0.2" "0.5") 

sampling_rates=("0.5")
algos=(
    "pFedMe" 
    "L2SGD"
    )
# DATA=("emnist" emnist_compon4ent "femnist" "cifar10" "cifar100")
DATA=(
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
    # "emnist" 
    # "emnist_alpha0.2" 
    # "emnist_alpha0.6"  
    # "emnist_alpha0.8" 
    # "emnist_pathologic_cl5" 
    # "emnist_pathologic_cl10" 
    # "emnist_pathologic_cl20" 
    "cifar100"
    )
get_prox_cmd
print_programs

algos=("FedAvg")
learner_rates=( "0.1")
get_ordinary_cmd

max_concurrent_processes=4

multi_run
commands=()

DATA=(
    # "femnist"  
    "cifar10"
    # "cifar100"
    )


# get_prox_cmd

# max_concurrent_processes=6
# print_programs

# multi_run

 