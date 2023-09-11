#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

<<<<<<< HEAD

source shell_experiments/run.sh
# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

# DATA=("cifar10_alpha0.8" )
multi_run(){
    pids=()
    for program in "${commands[@]}"; do
        while [ "$current_processes" -ge $max_concurrent_processes ]; do
            wait -n  # 等待任何子进程结束
            ((current_processes--))  # 子进程结束，计数减1
            echo current_processes  = "${current_processes}"
        done
        echo "$program"
        eval "$program" &
        # sleep 3
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


n_rounds=200
commands=()
current_processes=0

# DATA=("femnist" )


# sampling_rates=("0.5")
# max_concurrent_processes=1
# algos=("FedEM")
# get_ordinary_cmd

# multi_run


# sampling_rates=("0.25" )
# algos=("FedAvg")
# get_ordinary_cmd

# sampling_rates=("0.5")


# pre_rounds_list=("1")

# trans_list=("0.75")
# fuzzy_m_momentums=("0.8")
# fuzzy_m_schedulers=("cosine_annealing")
# measurements=("loss" "euclid")

# fuzzy_m_list=("1.5" "1.6"  "1.7" "1.8" "2" "2.2" "2.4")

# get_fuzzy_cmd

# sampling_rates=("0.25")

# pre_rounds_list=("1")

# trans_list=("0.75")
# fuzzy_m_momentums=("0.8")
# fuzzy_m_schedulers=("constant")
# measurements=("loss" "euclid")

# fuzzy_m_list=("1.5" "1.6"  "1.7" "1.8" "2" "2.2" "2.4")

# get_fuzzy_cmd

# echo commands length: ${#commands[@]}
# for program in "${commands[@]}"; do
#     echo "$program"
# done
# echo 

# max_concurrent_processes=3

# multi_run

DATA=("emnist_pathologic_cl20")



sampling_rates=("0.5")

pre_rounds_list=("1")

trans_list=("0.75")
fuzzy_m_momentums=("0.8")
fuzzy_m_schedulers=("constant")
measurements=("loss" "euclid")

fuzzy_m_list=("1.5" "1.6"  "1.7" "1.8" "2" "2.2" "2.4")

get_fuzzy_cmd

algos=("FedAvg")
get_ordinary_cmd

echo commands length: ${#commands[@]}
for program in "${commands[@]}"; do
    echo "$program"
done
echo 

max_concurrent_processes=4

multi_run








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
DATA=( "emnist_alpha0.6"  )
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


fuzzy_m_momentums=("0.8")
trans_list=("0.75")

# fuzzy_m_list=( "1.4" "1.5" "1.6" "1.7" "1.8" "2.2")
fuzzy_m_list=( "1.4" "2.2")
measurements=("loss" "euclid")
fuzzy_m_schedulers=("cosine_annealing" "constant")
max_concurrent_processes=2
get_fuzzy_cmd

print_programs

multi_run
>>>>>>> 4088e3c (.)
