#!/bin/bash

# 获取参数
REF_RTTM="/home3/yihao/Research/Dataset/third_dihard_challenge_eval/data/rttm/DH_EVAL_0014.rttm"
SYS_RTTM="/home3/yihao/Research/Code/VBx/sample/sep/DH_EVAL_0014.rttm"
UEM_FILE="/home3/yihao/Research/Dataset/third_dihard_challenge_eval/data/uem/DH_EVAL_0014.uem"
OUTPUT_FILE="result_uem_DH14"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Score
$DIR/dscore/score.py -u $UEM_FILE -r $REF_RTTM -s $SYS_RTTM > $OUTPUT_FILE