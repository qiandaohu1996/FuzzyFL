#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh
# DATA=("emnist" "emnist_component4" "femnist" "cifar10" "cifar100")
# DATA=("emnist_component4" "femnist" "cifar10")
DATA=("emnist20")
# DATA=("cifar10_pathologic_cl3")


alphas=("0.5")

run_agfl
