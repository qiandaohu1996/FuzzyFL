#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh
# DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("emnist")

sampling_rates=("0.2")

# DATA=("emnist" emnist_compon4ent "femnist" "cifar10" "cifar100")
DATA=("cifar10")

mus=("0.1") 
  
run_prox  

 