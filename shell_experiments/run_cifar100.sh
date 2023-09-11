#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

<<<<<<< HEAD

source shell_experiments/run.sh
# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

# DATA=("cifar10_alpha0.8" )
=======
source shell_experiments/run.sh
>>>>>>> 4088e3c (.)

n_rounds=200

current_processes=0
<<<<<<< HEAD
max_concurrent_processes=3

sampling_rates=( "0.5")

commands=()

DATA=("cifar100")

pre_rounds_list=("50")

trans_list=("0.75")
fuzzy_m_momentums=("0.8")
fuzzy_m_schedulers=("constant")
measurements=("loss" "euclid" )

fuzzy_m_list=("1.5" "1.8" "1.6" "1.7" "2.0")

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
=======
pre_rounds_list=("1")

sampling_rates=( "0.5")
# sampling_rates=( "1")
commands=()
 
lr=0.1
 
# DATA=("emnist_pathologic_cl5" )
# DATA=("cifar100")
DATA=("cifar100")
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
 
commands=()
n_clusters=3
top=3
fuzzy_m_momentums=("0.8")
trans_list=("0.75")

fuzzy_m_list=("1.5" "1.6" "1.7" "1.8"  "2" "2.2")
measurements=("loss" "euclid")
fuzzy_m_schedulers=("cosine_annealing" "constant")
get_fuzzy_cmd

print_programs
max_concurrent_processes=3
multi_run

commands=()
algos=("FedAvg" "FedEM")
get_ordinary_cmd
print_programs
max_concurrent_processes=1
multi_run

# commands=()

# DATA=("feminst")

# commands=()

# alogs=("FedEM" "FedAvg")
# get_ordinary_cmd

# print_programs
# max_concurrent_processes=1
# multi_run
>>>>>>> 4088e3c (.)
