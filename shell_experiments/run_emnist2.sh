#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh

n_rounds=200

current_processes=0
pre_rounds_list=("1")

sampling_rates=( "0.5")
# sampling_rates=( "1")
n_clusters=10
top=5
commands=()

lr=0.05

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
    "emnist" 
    "emnist_alpha0.2" 
    "emnist_alpha0.6"  
    "emnist_alpha0.8" 
    # "emnist_pathologic_cl5" 
    # "emnist_pathologic_cl10" 
    # "emnist_pathologic_cl20" 
    )

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

learner_rates=("0.02" "0.05" "0.1")
# learner_rates=("0.1")

# algos=("FedAvg")
# get_ordinary_cmd
# fuzzy_m_momentums=("0.8")
# trans_list=("0.75")

# n_clusters=5
# top=5


# fuzzy_m_list=( "1.6" )
# measurements=( "euclid")
# fuzzy_m_schedulers=( "constant")
# 
# fuzzy_m_list=("1.5" "1.7" "1.8" "2")
# fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2" "2.2")
# fuzzy_m_list=("2.2" )
# fuzzy_m_list=("1.6")
# fuzzy_m_schedulers=("constant")
# fuzzy_m_schedulers=( "constant")
# fuzzy_m_schedulers=( "cosine_annealing")

# measurements=("euclid")
# get_fuzzy_cmd 
# trans_list=("0.9")
# measurements=("loss")


# n_clusters=5
# top=5
# get_fuzzy_cmd 
# max_concurrent_processes=5

# print_programs
# multi_run
# commands=()

# fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2" "2.2")

# n_clusters=5
# top=5
# get_fuzzy_cmd 
# max_concurrent_processes=9

# print_programs
# multi_run

# commands=()
# learner_rates=( "0.02")
# DATA=(
    # "emnist_c10_alpha0.2"
#     "emnist_c10_alpha0.3"
#     "emnist_c10_alpha0.4"
#     "emnist_c10_alpha0.5"
#     "emnist_c10_alpha0.6"
# )
algos=("FedEM")
n_learners=3
learner_rates=("0.02" "0.05" "0.1")
# DATA=(
#     "emnist_alpha0.4"
# )
get_ordinary_cmd

max_concurrent_processes=1
print_programs
multi_run

 