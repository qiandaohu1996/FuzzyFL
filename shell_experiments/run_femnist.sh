#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

<<<<<<< HEAD

source shell_experiments/run.sh
# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

# DATA=("cifar10_alpha0.8" )

DATA=("femnist" )
n_rounds=200

current_processes=0
max_concurrent_processes=1

sampling_rates=( "0.5")

commands=()

# algos=("FedEM")
# get_ordinary_cmd

pre_rounds_list=("1")

fuzzy_m_schedulers=("cosine_annealing")

fuzzy_m_momentums=("0.8")

fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2" "2.2" "2.4")
trans_list=("0.75")
measurements=("loss" )
get_fuzzy_cmd

# fuzzy_m_momentums=("0.8")

# fuzzy_m_list=("1.5")
# trans_list=("0.75")
# measurements=("euclid")
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
        echo
        echo "$program"
        eval "$program" &
        pids+=($!)  # 存储子进程的PID
        echo process "$!" start
        ((current_processes++))
        echo current_processes  = "${current_processes}"
=======
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
>>>>>>> 4088e3c (.)
        sleep 10
    done

    # 等待剩余的子进程结束
<<<<<<< HEAD
    for pid in "${pids[@]}"; do
        wait "$pid"
        echo process "$pid" end
    done
    echo current_processes  = "${current_processes}"

    echo "All programs have finished execution."
}

=======
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
>>>>>>> 4088e3c (.)
multi_run