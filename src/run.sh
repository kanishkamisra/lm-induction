#!/bin/sh

declare -a models=(axxl-property rl-property bl-property)

for model in ${models[@]}
do
    python taxonomic_generalization.py -m ${model} -d cuda:0
done