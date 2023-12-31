#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh

n_rounds=200

current_processes=0
pre_rounds_list=("1")
learner_rates=("0.02" "0.05" )
sampling_rates=( "0.5")
# sampling_rates=( "1")
n_clusters=10
top=5
commands=()


DATA=(
    "emnist" 
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
    "emnist_alpha0.2" 
    "emnist_alpha0.6"  
    "emnist_alpha0.8" 
    "emnist_pathologic_cl5" 
    "emnist_pathologic_cl10" 
    "emnist_pathologic_cl20" 
    "cifar10"
    "cifar100"
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
        sleep 5
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
 
algos=("FedAvg")
get_ordinary_cmd
 
max_concurrent_processes=3

print_programs
multi_run
 