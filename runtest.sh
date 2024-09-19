#!/usr/bin/env bash

cmd="/home3/theanhtran/slurm.pl --quiet" #  --nodelist=node06 --gpu 1

#. /home3/theanhtran/env/anaconda/etc/profile.d/conda.sh
conda activate VBx

$cmd log/track.log \
./DIHARD2_run.sh