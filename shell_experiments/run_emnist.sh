#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091


source shell_experiments/run.sh
# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

DATA=("emnist_alpha0.8" )

n_rounds=200

current_processes=0
max_concurrent_processes=2

sampling_rates=( "0.5")

commands=()

lr=0.05
algos=("FedAvg")
get_ordinary_cmd
lr=0.1

pre_rounds_list=("50")

fuzzy_m_schedulers=("constant")

fuzzy_m_momentums=("0.8" "0.5")
fuzzy_m_list=("1.8" "2")
trans_list=("0.75")
measurements=("loss")
get_fuzzy_cmd

fuzzy_m_momentums=("0.5")
fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2")
trans_list=("0.75")
measurements=("loss" "euclid")
get_fuzzy_cmd

fuzzy_m_momentums=("0.5" "0.8")
fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2")
trans_list=("0.75")
measurements=("loss" "euclid")
fuzzy_m_schedulers=("cosine_annealing")
get_fuzzy_cmd

DATA=("emnist")

fuzzy_m_momentums=("0.8")
fuzzy_m_list=( "1.7" "1.8")
trans_list=("0.75")
measurements=("loss" )
fuzzy_m_schedulers=("constant")
get_fuzzy_cmd

fuzzy_m_momentums=("0.5")
fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2")
trans_list=("0.75")
measurements=("loss")
fuzzy_m_schedulers=("constant" "cosine_annealing")
get_fuzzy_cmd


DATA=("emnist_pathologic_cl20" "emnist_pathologic_cl5" "emnist_c5" "emnist_alpha0.2")

fuzzy_m_momentums=("0.8")

fuzzy_m_list=("1.7")
trans_list=("0.75")
measurements=("loss"  "euclid")
fuzzy_m_schedulers=("constant")
get_fuzzy_cmd

fuzzy_m_momentums=("0.5")

fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2")
trans_list=("0.75")
measurements=("loss"  "euclid")
fuzzy_m_schedulers=("constant")
get_fuzzy_cmd


for program in "${commands[@]}"; do
    echo "$program"
done

multi_run(){
    pids=()
    for program in "${commands[@]}"; do
        while [ $current_processes -ge $max_concurrent_processes ]; do
            wait -n  # 等待任何子进程结束
            ((current_processes--))  # 子进程结束，计数减1
            echo current_processes  = "${current_processes}"
        done
        echo "$program"
        eval "$program" &
        pids+=($!)  # 存储子进程的PID
        echo process "$!" start
        ((current_processes++))
        echo current_processes  = "${current_processes}"
    done

    # 等待剩余的子进程结束
    for pid in "${pids[@]}"; do
        wait "$pid"
        echo process "$pid" end
    done
    echo current_processes  = "${current_processes}"

    echo "All programs have finished execution."
}

max_concurrent_processes=3

multi_run