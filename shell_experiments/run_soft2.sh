#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh

n_rounds=200
commands=()

pre_rounds_list=("10")

sampling_rates=( "0.5")
n_clusters=3
mus=("0" "0.01" )
# mus=("0.01"  "0.1" "0.001"  )
tau=1
# learner_rates=("0.02" "0.05" "0.1")
learner_rates=("0.05")

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
    )

get_soft_cmd
max_concurrent_processes=1
print_programs

multi_run

 