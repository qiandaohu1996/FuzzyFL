#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh
# DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("emnist")

# DATA=("emnist" "emnist_compon4ent" "femnist" "cifar10" "cifar100")
DATA=("cifar10" )
# dataset="shakespeare"

sampling_rates=("0.2")

DATA=("cifar10")

run_clustered

 
