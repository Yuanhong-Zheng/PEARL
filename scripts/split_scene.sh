#!/bin/bash

# 项目根目录（PEARL）
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 设置视频文件夹和 Python 脚本路径（相对项目根目录）
VIDEO_DIR="${PROJECT_ROOT}/data/video-level"
SCRIPT_PATH="${PROJECT_ROOT}/video_scene_splitter.py"

# 遍历文件夹下所有mp4文件
for video_file in "$VIDEO_DIR"/*.mp4; do
    # 检查文件是否存在（避免没有mp4文件时的错误）
    if [ -f "$video_file" ]; then
        echo "正在处理: $video_file"
        python "$SCRIPT_PATH" --video_path "$video_file"
        echo "完成处理: $video_file"
        echo "-----------------------------------"
    fi
done

echo "所有视频处理完成！"