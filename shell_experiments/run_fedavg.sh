#!/usr/bin/env bash
# shellcheck disable=SC2034
# shellcheck disable=SC1091

source shell_experiments/run.sh
# DATA=("emnist" "femnist" "cifar10" "cifar100")
# DATA=("cifar10")
# DATA=("cifar10_alpha0.8")
DATA=("emnist20")

sampling_rates=("0.5")

run_avg
