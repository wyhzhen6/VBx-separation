import os
from pyannote.audio import Pipeline
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def parse_rttm(file_path):
    segments = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 10:
                continue
            speaker = parts[7]
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            
            if speaker not in segments:
                segments[speaker] = []
            segments[speaker].append((start_time, end_time))
    return segments

def plot_segments(segments, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20.colors  # 用于分配不同颜色
    y_labels = []
    
    for idx, (speaker, times) in enumerate(segments.items()):
        y_labels.append(speaker)
        for start, end in times:
            ax.add_patch(
                patches.Rectangle(
                    (start, idx - 0.4),  # (x, y)
                    end - start,  # width
                    0.8,  # height
                    color=colors[idx % len(colors)]
                )
            )
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time (s)')
    ax.set_title('Speaker Segments Over Time')
    ax.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_waveform(data, sample_rate, title, output_path):
    times = np.arange(len(data)) / float(sample_rate)
    plt.figure(figsize=(12, 4))
    plt.plot(times, data)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def perform_separation(input_folder, output_folder, token):
    # 初始化语音分离管道
    pipeline = Pipeline.from_pretrained(
        "pyannote/speech-separation-ami-1.0",
        use_auth_token=token
    )

    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):  # 仅处理 .wav 文件
            input_file_path = os.path.join(input_folder, filename)
            output_rttm_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".rttm")
            output_rttm_image = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
    
            # 读取输入音频
            sample_rate, input_data = scipy.io.wavfile.read(input_file_path)

            # 绘制输入音频波形
            input_waveform_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_input_waveform.png")
            plot_waveform(input_data, sample_rate, "Input Audio Waveform", input_waveform_path)

            # 执行语音分离
            diarization, sources = pipeline(input_file_path)

            # 保存分离结果到 RTTM 文件
            with open(output_rttm_path, 'w') as rttm:
                diarization.write_rttm(rttm)

            #提取说话者的说话区段并绘图
            #plot_segments(parse_rttm(output_rttm_path), output_rttm_image)

            # 检查 sources.data 的形状以确定有多少个说话者
            num_speakers = sources.data.shape[1]
            #print(f"Number of speakers detected: {num_speakers}")

            # 保存每个扬声器的音频到单独的文件中
            for s in range(num_speakers):
                speaker = diarization.labels()[s]
                output_speaker_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_{speaker}.wav')
                #output_audio_data_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_{speaker}_audio_data.txt')
                output_waveform_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_{speaker}_waveform.png')
                
                # 获取音频数据并检查数据类型
                audio_data = sources.data[:, s]
                #np.savetxt(output_audio_data_path, audio_data, fmt='%f')
                
                # 写入音频文件并确保格式正确
                try:
                    #scipy.io.wavfile.write(output_speaker_path, 16000,audio_data )
                    scipy.io.wavfile.write(output_speaker_path, 16000, audio_data.astype('float32'))
                    # 绘制分离后音频波形
                    plot_waveform(audio_data, 16000, f"Speaker {speaker} Audio Waveform", output_waveform_path)

                except Exception as e:
                    print(f"Error saving audio for speaker {speaker} to {output_speaker_path}: {e}")
            
            #print(f"Processed {input_file_path}, results saved to {output_rttm_path} and speaker files in {output_folder}")

if __name__ == "__main__":
    # 指定输入文件夹和输出文件夹路径
    input_folder = "/home3/yihao/Research/Code/VBx/sample/audio"
    output_folder = "/home3/yihao/Research/Code/VBx/sample/sep"
    
    huggingface_token = "**"

    perform_separation(input_folder, output_folder, huggingface_token)
    
