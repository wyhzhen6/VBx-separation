#!/usr/bin/env bash

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"



dir=exp
mkdir -p $dir

wav_dir=example/audios/dihard
lab_dir=example/vad
gt_dir=example/rttm

ark_dir=$dir/ark
seg_dir=$dir/seg
rttm_dir=$dir/rttm

mkdir -p $ark_dir
mkdir -p $seg_dir
mkdir -p $rttm_dir  

separator=none  # none, pyannote, nass


for audio in $(ls example/audios/dihard)
do      
    # 去处后缀，得到key
    filename=$(echo "${audio}" | cut -f 1 -d '.')
    

    echo "Processing VAD: $filename.wav"
    python VBx/utils/vad.py \
        --input_file $wav_dir/$filename.wav \
        --output_file $lab_dir/$filename.lab 
    echo "End VAD: $filename.wav"
    

    echo "Processing Separation and X-vector: $filename"
    echo ${filename} > $dir/list.txt
    python VBx/utils/predict.py \
        --separator $separator \
        --in-file-list $dir/list.txt \
        --in-lab-dir $lab_dir \
        --in-wav-dir $wav_dir \
        --out-ark-fn $ark_dir/${filename}.ark \
        --out-seg-fn $seg_dir/${filename}.seg \
        --weights VBx/models/ResNet101_16kHz/nnet/final.onnx \
        --backend onnx \
        --tmp-dir $dir 
    echo "End Separation and x-vector: $filename"


    echo "Processing Clustering and AHC: $filename"
    python VBx/utils/vbhmm.py \
        --init AHC+VB \
        --out-rttm-dir $rttm_dir \
        --xvec-ark-file $ark_dir/${filename}.ark \
        --segments-file $seg_dir/${filename}.seg \
        --xvec-transform VBx/models/ResNet101_16kHz/transform.h5 \
        --plda-file VBx/models/ResNet101_16kHz/plda \
        --threshold -0.015 \
        --lda-dim 128 \
        --Fa 0.3 \
        --Fb 17 \
        --loopP 0.99
     echo "Processing Clustering and AHC: $filename"


    if [ -f example/rttm/${filename}.rttm ]
    then
        # run dscore
        python dscore/score.py -r $gt_dir/${filename}.rttm -s $rttm_dir/${filename}.rttm --collar 0.25 --ignore_overlaps
    fi
    
done