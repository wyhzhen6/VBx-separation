import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def merge_rttm_files(rttm_files):
    merged_rttm = []
    for idx, rttm_file in enumerate(rttm_files):
        with open(rttm_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                parts.append(str(idx + 1))  # 添加文件索引
                merged_line = ' '.join(parts) + '\n'
                merged_rttm.append(merged_line)
    return merged_rttm

def save_merged_rttm(merged_rttm, output_file):
    with open(output_file, 'w') as file:
        file.writelines(merged_rttm)

def parse_rttm(file_path):
    segments = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 11:
                continue
            speaker_id = parts[7]
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            file_index = parts[10]
            speaker_label = f"{file_index}-{speaker_id}"
            
            if speaker_label not in segments:
                segments[speaker_label] = []
            segments[speaker_label].append((start_time, end_time))
    return segments

def plot_segments(segments, total_duration, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20.colors  # 用于分配不同颜色
    y_labels = list(segments.keys())
    
    for idx, (speaker, times) in enumerate(segments.items()):
        for start, end in times:
            ax.add_patch(
                patches.Rectangle(
                    (start, idx - 0.4),  # (x, y)
                    end - start,  # width
                    0.4,  # height
                    color=colors[idx % len(colors)]
                )
            )
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.8, len(y_labels) - 0.5)  # 调整 Y 轴范围
    ax.set_xlim(0, total_duration)
    ax.set_xlabel('Time (s)')
    ax.set_title('Speaker Segments Over Time')
    ax.grid(True)
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    rttm_files = ['/home3/yihao/Research/Code/VBx/exp/DH_EVAL_0014_SPEAKER_00.rttm', '/home3/yihao/Research/Code/VBx/exp/DH_EVAL_0014_SPEAKER_01.rttm', '/home3/yihao/Research/Code/VBx/exp/DH_EVAL_0014_SPEAKER_02.rttm']
    merged_rttm = merge_rttm_files(rttm_files)
    merged_rttm_file = 'merged-dh.rttm'
    output_image = "viewrttm-dh.png"
    save_merged_rttm(merged_rttm, merged_rttm_file)
    total_duration = 60  # 总时长为 60 秒
    
    segments = parse_rttm(merged_rttm_file)
    plot_segments(segments, total_duration, output_image)
    #print(f"Saved plot to {output_image}")
