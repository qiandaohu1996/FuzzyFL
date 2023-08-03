#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh
# DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("emnist")

sampling_rates=("0.2")

DATA=("cifar10")

comm_probs=("0.2")  
mus=("0.2" ) 
run_l2gd 
 

