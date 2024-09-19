#!/usr/bin/env bash

cmd="/home3/yihao/slurm.pl --quiet" #  --nodelist=node06 --gpu 1

conda init
source ~/.bashrc
conda activate VBx

$cmd sep.log \
./run_sep.sh
