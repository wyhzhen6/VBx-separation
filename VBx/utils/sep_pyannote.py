import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pyannote.audio import Pipeline
import scipy.io.wavfile

def separation_pyannote(input_file):

    # 初始化语音分离管道
    pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        use_auth_token="xxxxx",
    )

    if input_file.endswith(".wav"):  # 仅处理 .wav 文件

        # 执行语音分离
        _, sources = pipeline(input_file)

        # 检查 sources.data 的形状以确定有多少个说话者
        num_speakers = sources.data.shape[1]
        print(f"Number of speakers detected: {num_speakers}")

        return sources.data
