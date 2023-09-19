#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh

n_rounds=200

current_processes=0
pre_rounds_list=("1")

sampling_rates=( "0.5")
# sampling_rates=( "1")

commands=()
 
lr=0.1
 
# DATA=("emnist_pathologic_cl5" )
DATA=(
    "seed23456_emnist" 
    "seed23456_emnist_alpha0.2"
    "seed23456_emnist_alpha0.6" 
    "seed23456_emnist_alpha0.8" 
    "seed23456_emnist_pathologic_cl5"
    "seed23456_emnist_pathologic_cl10"
    "seed23456_emnist_pathologic_cl20"
)
# DATA=( "emnist_alpha0.6"  )
# DATA=("emnist"  )
# DATA=( "emnist_pathologic_cl20" "emnist")

multi_run() { 
    for cmd in "${commands[@]}"; do
        while [ $current_processes -ge $max_concurrent_processes ]; do
            wait -n  # 等待任何子进程结束
            ((current_processes--))  # 子进程结束，计数减1
            echo current_processes = "${current_processes}"
        done
        eval "$cmd" &
        echo 
        # echo "$cmd" 
        echo process "$!" start
        ((current_processes++))
        echo current_processes = "${current_processes}"
        sleep 10
    done

    # 等待剩余的子进程结束
    while [ $current_processes -gt 0 ]; do
        wait -n
        ((current_processes--))
        echo current_processes = "${current_processes}"
    done

    echo "All programs have finished execution."
commands=()

}

# print_programs
# multi_run
# commands=()

# algos=("FedEM" "FedAvg")
# algos=("FedAvg")
# bz=128
# local_steps=2

# get_ordinary_cmd

# max_concurrent_processes=3

# print_programs
# multi_run 

fuzzy_m_momentums=("0.8")
trans_list=("0.75")

# fuzzy_m_list=( "1.5"  "1.6"  "1.7"  "1.8"  "2"  "2.2" )
fuzzy_m_list=(  "2.2" )
# measurements=("loss" "euclid")
measurements=( "euclid")
fuzzy_m_schedulers=("cosine_annealing")
# max_concurrent_processes=12
# get_fuzzy_cmd
 
# print_programs

# multi_run
commands=()
# DATA=("femnist")

# algos=("FedEM" "FedAvg")
# get_ordinary_cmd
# print_programs
# max_concurrent_processes=2
# multi_run


# DATA=("feminst")
DATA=("seed23456_femnist")
get_fuzzy_cmd

# print_programs
# max_concurrent_processes=3
# multi_run
# commands=()

# algos=("FedEM" "FedAvg")
# get_ordinary_cmd

print_programs
max_concurrent_processes=2
multi_run