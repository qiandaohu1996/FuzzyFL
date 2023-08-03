#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091


source shell_experiments/run.sh
# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")
# DATA=("emnist_pathologic_cl20" )

# DATA=("cifar10_alpha0.8" )

n_rounds=200



sampling_rates=( "0.5")

commands=()

DATA=("cifar100")

# mus=("0.05" "0.1"  "0.5")
# algos=("FedProx" "L2SGD" "pFedMe")
# get_prox_cmd

# alphas=("0.5" "0.25")
# get_apfl_cmd

sampling_rates=( "0.5")
pre_rounds_list=("50")

trans_list=("0.75" "0.5")
fuzzy_m_momentums=("0.8")
fuzzy_m_schedulers=("constant")
measurements=("loss" "euclid" )

fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2.0")

get_fuzzy_cmd

for program in "${commands[@]}"; do
    echo "$program"
done

current_processes=0

multi_run(){
    pids=()
    # 存储已经开始的命令和它们在commands数组中的索引
    started_commands=()
    started_indices=()
    for index in "${!commands[@]}"; do
        program=${commands[index]}
        echo "$program"
        while true; do
            # 使用nvidia-smi获取显卡剩余显存
            free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk '{ print int($1/2048) }')
            high_memory=16
            low_memory=8
            # 如果剩余显存大于4G, 则启动新进程
            if (( free_memory > high_memory )); then
                echo "$program"
                eval "$program" &
                pids+=($!)  # 存储子进程的PID
                echo "${pids[@]}"
                started_commands+=("$program")  # 存储已经开始的命令
                started_indices+=("$index")  # 存储已经开始的命令的索引
                echo "${started_commands[@]}"
                echo "${started_indices[@]}"
                echo process "$!" start
                break
            fi
            sleep 1m
            # 如果剩余显存小于2G, 则结束最后一个进程
            if (( free_memory < low_memory && ${#pids[@]} > 0 )); then
                # 获取最后一个进程的PID
                last_pid=${pids[${#pids[@]}-1]}
                last_command=${started_commands[${#started_commands[@]}-1]}
                last_index=${started_indices[${#started_indices[@]}-1]}
                kill -TERM "$last_pid"
                echo process "$last_pid" stopped
                high_memory+=2048
                # 移除最后一个进程的信息
                unset "pids[${#pids[@]}-1]"
                unset "started_commands[${#started_commands[@]}-1]"
                unset "started_indices[${#started_indices[@]}-1]"

                # 将停止的命令重新加入到commands数组的开始
                commands=("$last_command" "${commands[@]}")

                # 如果当前命令是被停止的命令，更新索引以避免重复执行
                if (( index == last_index )); then
                    ((index--))
                fi
            fi
            echo $high_memory
            # 等待一段时间再检查
            sleep 5m
        done
    done

    # 等待剩余的子进程结束
    for pid in "${pids[@]}"; do
        wait "$pid"
        echo process "$pid" end
    done

    echo "All programs have finished execution."
}

multi_run