import sys
import os
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.core import Segment
from pyannote.audio import Model


def perform_vad(input_folder, output_folder):
    # 初始化 VoiceActivityDetection 管道
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token="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")
    pipeline = VoiceActivityDetection(segmentation=model)
    # 定义超参数
    HYPER_PARAMETERS = {
        #"onset": 0.5, "offset": 0.5,
        "min_duration_on": 0.0,  # 移除短于此时间的语音段
        "min_duration_off": 0.0  # 填补短于此时间的非语音段
    }

    # 实例化管道
    pipeline.instantiate(HYPER_PARAMETERS)

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  # 仅处理 .wav 文件
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".lab")

            # 执行 VAD
            vad = pipeline(input_file_path)

            # 保存 VAD 结果到文件
            with open(output_file_path, 'w') as f:
                for speech_turn in vad.get_timeline():
                    f.write(f"{speech_turn.start:.3f}\t{speech_turn.end:.3f}\tspeech\n")
            
            #print(f"Processed {input_file_path}, results saved to {output_file_path}")

if __name__ == "__main__":
    # 指定输入文件夹和输出文件夹路径
    input_folder = "/home3/yihao/Research/Code/VBx/sample/audio"
    output_folder = "/home3/yihao/Research/Code/VBx/sample/vad"
    
    perform_vad(input_folder, output_folder)
