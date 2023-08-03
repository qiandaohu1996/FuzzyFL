#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh


# DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("emnist" "emnist_component4" "femnist" "cifar10" "cifar100" "cifar100_s0.25")

# DATA=("emnist" "emnist_c4" "femnist" "emnist_pathologic_cl20"   "cifar100_s0.25" "cifar10_n50"   "cifar100_n20")

# DATA=("emnist_pathologic_cl20" "emnist_component4")
DATA=("cifar10")
# DATA=( "femnist179" )
 
sampling_rates=("0.2")
mus=("1" "10" "100" "0.1" "0.01") 

run_pfedme
 
 
