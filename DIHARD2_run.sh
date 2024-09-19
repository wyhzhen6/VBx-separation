#!/bin/bash

INSTRUCTION="xvectors"
METHOD="AHC+VB" # AHC or AHC+VB

exp_dir="/home3/yihao/Research/Code/VBx/exp-dihard3" # output experiment directory
xvec_dir="/home3/yihao/Research/Code/VBx/xvec" # output xvectors directory
WAV_DIR="/home3/yihao/Research/Dataset/third_dihard_challenge_eval/data/wav" # wav files directory
FILE_LIST="/home3/yihao/Research/Dataset/list.txt" # txt list of files to process
LAB_DIR="/home3/yihao/Research/Code/VBx/pyaanote_vad/dihard3" # lab files directory with VAD segments
RTTM_DIR="/home3/yihao/Research/Dataset/third_dihard_challenge_eval/data/rttm" # reference rttm files directory
UEMCORE_DIR="/home3/yihao/Research/Dataset/third_dihard_challenge_eval/data/uem_scoring/core/all.uem" #un-partitioned evaluation map (UEM) files directory
UEMFULL_DIR="/home3/yihao/Research/Dataset/third_dihard_challenge_eval/data/uem_scoring/full/all.uem"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


if [[ $INSTRUCTION = "xvectors" ]]; then
	WEIGHTS_DIR=$DIR/VBx/models/ResNet101_16kHz/nnet
	if [ ! -f $WEIGHTS_DIR/raw_81.pth ]; then
	    cat $WEIGHTS_DIR/raw_81.pth.zip.part* > $WEIGHTS_DIR/unsplit_raw_81.pth.zip
		unzip $WEIGHTS_DIR/unsplit_raw_81.pth.zip -d $WEIGHTS_DIR/
	fi

	WEIGHTS=$DIR/VBx/models/ResNet101_16kHz/nnet/raw_81.pth
	EXTRACT_SCRIPT=$DIR/VBx/extract.sh
	DEVICE=cpu

	mkdir -p $xvec_dir
	$EXTRACT_SCRIPT ResNet101 $WEIGHTS $WAV_DIR $LAB_DIR $FILE_LIST $xvec_dir $DEVICE

	# Replace this to submit jobs to a grid engine
	bash $xvec_dir/xv_task
fi


BACKEND_DIR=$DIR/VBx/models/ResNet101_16kHz
if [[ $INSTRUCTION = "diarization" ]]; then
    #echo "Processing with METHOD=$METHOD"
	TASKFILE=$exp_dir/diar_"$METHOD"_task
	#echo "TASKFILE=$TASKFILE"
	OUTFILE=$exp_dir/diar_"$METHOD"_out
	#echo "OUTFILE=$OUTFILE"
	rm -f $TASKFILE $OUTFILE
	mkdir -p $exp_dir/lists

	thr=-0.015 #初始化的阈值
	smooth=7.0 #平滑参数
	lda_dim=128 #LDA（线性判别分析）维度
	Fa=0.2
	Fb=6
	loopP=0.35
	#变分贝叶斯的其他参数
	OUT_DIR=$exp_dir/out_dir_"$METHOD"
	#每次执行完删掉exp-dihard3/out_dir_AHC+VB
	$DIR/dscore/score.py -u $UEMFULL_DIR -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_uem_full
	if [[ ! -d $OUT_DIR ]]; then
		mkdir -p $OUT_DIR
		while IFS= read -r line; do
			grep $line $FILE_LIST > $exp_dir/lists/$line".txt"
			python3="unset PYTHONPATH ; unset PYTHONHOME ; export PATH=\"/mnt/matylda5/iplchot/python_public/anaconda3/bin:$PATH\""
			echo "$python3 ; python $DIR/VBx/vbhmm.py --init $METHOD --out-rttm-dir $OUT_DIR/rttms --xvec-ark-file $xvec_dir/xvectors/$line.ark --segments-file $xvec_dir/segments/$line --plda-file $BACKEND_DIR/plda --xvec-transform $BACKEND_DIR/transform.h5 --threshold $thr --init-smoothing $smooth --lda-dim $lda_dim --Fa $Fa --Fb $Fb --loopP $loopP" >> $TASKFILE
		done < $FILE_LIST
		bash $TASKFILE > $OUTFILE

		# Score
		cat $OUT_DIR/rttms/*.rttm > $OUT_DIR/sys.rttm
		cat $RTTM_DIR/*.rttm > $OUT_DIR/ref.rttm
		$DIR/dscore/score.py --collar 0.25 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_fair
		$DIR/dscore/score.py --collar 0.0 -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_full
		$DIR/dscore/score.py -u $UEMCORE_DIR -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_uem_core
		$DIR/dscore/score.py -u $UEMFULL_DIR -r $OUT_DIR/ref.rttm -s $OUT_DIR/sys.rttm > $OUT_DIR/result_uem_full
	fi
fi