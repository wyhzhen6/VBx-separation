import sys
import os
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment
from pyannote.audio import Model
import argparse
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def perform_vad(input_file, output_file):
    # 初始化 VoiceActivityDetection 管道/
    model = Model.from_pretrained("pyannote/segmentation-3.0")
    pipeline = VoiceActivityDetection(segmentation=model)
    # 定义超参数
    HYPER_PARAMETERS = {
        #"onset": 0.5, "offset": 0.5,
        "min_duration_on": 0.0,  # 移除短于此时间的语音段
        "min_duration_off": 0.0  # 填补短于此时间的非语音段
    }

    # 实例化管道
    pipeline.instantiate(HYPER_PARAMETERS)

    filename = input_file

    if filename.endswith(".wav"):  # 仅处理 .wav 文件
        vad = pipeline(filename)

        # 保存 VAD 结果到文件
        with open(output_file, 'w') as f:
            for speech_turn in vad.get_timeline():
                f.write(f"{speech_turn.start:.3f}\t{speech_turn.end:.3f}\tspeech\n")

    else: 
        raise ValueError("Input file must be a .wav file")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, type=str, help='input_file')
    parser.add_argument('--output_file', required=True, type=str, help='output_file')
    args = parser.parse_args()

    
    perform_vad(args.input_file, args.output_file)