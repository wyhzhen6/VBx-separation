#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# @Authors: Federico Landini
# @Emails: landini@fit.vutbr.cz

MODEL=$1
WEIGHTS=$2
WAV_DIR=$3
LAB_DIR=$4
FILE_LIST=$5
OUT_DIR=$6
DEVICE=$7

EMBED_DIM=256
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

#这部分代码创建输出目录，并初始化任务文件 ($TASKFILE) 和相关子目录
mkdir -pv $OUT_DIR

TASKFILE=$OUT_DIR/xv_task
rm -f $TASKFILE

mkdir -p $OUT_DIR/lists $OUT_DIR/xvectors $OUT_DIR/segments
#对文件列表中的每个条目进行处理：
while IFS= read -r line; do
    #为当前条目创建必要的目录。
	mkdir -p "$(dirname $OUT_DIR/lists/$line)"
    #使用 grep 从文件列表中提取当前条目，并保存到 $OUT_DIR/lists/$line.txt。
    grep $line $FILE_LIST > $OUT_DIR/lists/$line".txt"
    #设置输出的ark文件路径和段文件路径。
    OUT_ARK_FILE=$OUT_DIR/xvectors/$line.ark
    OUT_SEG_FILE=$OUT_DIR/segments/$line
    mkdir -p "$(dirname $OUT_ARK_FILE)"
    mkdir -p "$(dirname $OUT_SEG_FILE)"
    #根据设备类型（gpu或cpu）构建Python命令行，并将其添加到任务文件 ($TASKFILE) 中
    if [[ "$DEVICE" == "gpu" ]]; then
    	echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --gpus=\$($DIR/free_gpu.sh) $MDL_WEIGHTS --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    else
    	echo "python $DIR/predict.py --seg-len 144 --seg-jump 24 --model $MODEL --weights $WEIGHTS --gpus= $MDL_WEIGHTS --ndim 64 --embed-dim $EMBED_DIM --in-file-list $OUT_DIR/lists/$line".txt" --in-lab-dir $LAB_DIR --in-wav-dir $WAV_DIR --out-ark-fn $OUT_ARK_FILE --out-seg-fn $OUT_SEG_FILE" >> $TASKFILE
    fi
    #任务执行过程中，每个任务会调用 predict.py 脚本来实际执行特征提取和x-vectors计算
done < $FILE_LIST
