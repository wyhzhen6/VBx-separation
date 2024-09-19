import os
from pyannote.audio import Pipeline
import scipy.io.wavfile

def perform_separation(input_folder, output_folder):
    # 初始化语音分离管道
    pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        use_auth_token="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"
    )

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  # 仅处理 .wav 文件
            input_file_path = os.path.join(input_folder, filename)
            output_rttm_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".rttm")

            # 执行语音分离
            diarization, sources = pipeline(input_file_path)

            # 保存分离结果到 RTTM 文件
            with open(output_rttm_path, 'w') as rttm:
                diarization.write_rttm(rttm)

            # 检查 sources.data 的形状以确定有多少个说话者
            num_speakers = sources.data.shape[1]
            print(f"Number of speakers detected: {num_speakers}")

            # 保存每个扬声器的音频到单独的文件中
            for s in range(num_speakers):
                speaker = diarization.labels()[s]
                output_speaker_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_{speaker}.wav')
                scipy.io.wavfile.write(output_speaker_path, 16000, sources.data[:, s])
            
            print(f"Processed {input_file_path}, results saved to {output_rttm_path} and speaker files in {output_folder}")


if __name__ == "__main__":
    # 指定输入文件夹和输出文件夹路径
    input_folder = "/home3/yihao/Research/Code/VBx/sample/audio"
    output_folder = "/home3/yihao/Research/Code/VBx/sample/sep"

    perform_separation(input_folder, output_folder)
