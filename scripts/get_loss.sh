#!/usr/bin/env bash

# This script extracts the best loss values for each epoch of an evaluated model,
# for the evaluation on the training datasubset and the validation datasubset.
# The respective 'train.log' file of the evaluated model is expected as "${1}",
# while the file where to write the extracted loss values is expected as "${2}".
# Example invocation:
#   user@machine $   ./get_loss.sh results/exp01/train.log results/exp01/loss.csv

# Previos command:
# grep "Best" $1 | grep drift | cut -d' ' -f10,15 > $2

# Updated command:
grep "Best" "${1}" | tr -s " " | cut -d " " -f 8,12 > "${2}"
