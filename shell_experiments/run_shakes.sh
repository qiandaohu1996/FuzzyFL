#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh

n_rounds=200
DATA=("shakespeare_s0.3" )
commands=()

pre_rounds_list=("1")
learner_rates=( "0.1")

sampling_rates=( "0.5" "0.75")

fuzzy_m_momentums=("0.8")
trans_list=("0.75")
n_clusters=3
top=3


algos=("FedAvg")

get_ordinary_cmd

mus=("0.05" "0.1" "0.2" "0.5") 

algos=("FedProx" "pFedMe")
get_prox_cmd
max_concurrent_processes=6

print_programs
multi_run
commands=()
fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2")

# trans_list=("0.9" "0.75")
measurements=("euclid" "loss")
fuzzy_m_schedulers=("constant" )
get_fuzzy_cmd 
# commands=()

# fuzzy_m_list=("1.5" "1.6" "1.7" "1.8" "2" "2.2")

# n_clusters=5
# top=5
# get_fuzzy_cmd 
max_concurrent_processes=4

print_programs
multi_run
 