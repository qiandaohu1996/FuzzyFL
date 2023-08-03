#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091


source shell_experiments/run.sh
# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

# DATA=("cifar10_alpha0.8" )

n_rounds=200

current_processes=0
max_concurrent_processes=5

sampling_rates=( "0.5" )

commands=()

DATA=("cifar10")

# lr=0.05
# algos=("FedAvg")
# get_ordinary_cmd

pre_rounds_list=("50")

fuzzy_m_schedulers=("constant")

fuzzy_m_momentums=("0.8" )
fuzzy_m_list=("2" "1.8" "1.7" "1.6" "1.5")
trans_list=("0.5")
measurements=("loss" "euclid")
get_fuzzy_cmd

# fuzzy_m_schedulers=("cosine_annealing")

# fuzzy_m_momentums=("0.8")
# fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2.0" "2.2")
# trans_list=("0.75")
# measurements=("loss" "euclid")
# get_fuzzy_cmd

# sampling_rates=("0.25" "0.75" "1")
# fuzzy_m_schedulers=("constant")

# fuzzy_m_momentums=("0.8" )
# fuzzy_m_list=("2" "1.8" "1.6" )
# trans_list=("0.75" )
# measurements=("loss" "euclid")
# get_fuzzy_cmd

echo commands length: ${#commands[@]}

for program in "${commands[@]}"; do
    echo "$program"
done
echo 
multi_run(){
    pids=()
    for program in "${commands[@]}"; do
        while [ $current_processes -ge $max_concurrent_processes ]; do
            wait -n  # 等待任何子进程结束
            ((current_processes--))  # 子进程结束，计数减1
            echo current_processes  = "${current_processes}"
        done
        eval "$program" &
        pids+=($!)  # 存储子进程的PID
        echo 
        echo "$program" process "$!" start
        ((current_processes++))
        echo current_processes  = "${current_processes}"
        sleep 20
    done

    # 等待剩余的子进程结束
    for pid in "${pids[@]}"; do
        wait "$pid"
        echo process "$pid" end
    done
    echo current_processes  = "${current_processes}"

    echo "All programs have finished execution."
}
max_concurrent_processes=2

multi_run