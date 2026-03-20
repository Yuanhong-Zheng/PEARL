#!/bin/bash

# Project root (PEARL)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

OUTPUT_DIR="${1:-${PROJECT_ROOT}/data/video-level/output_clips}"

# Configure the video directory and Python script path relative to the project root
VIDEO_DIR="${PROJECT_ROOT}/data/video-level/videos"
SCRIPT_PATH="${PROJECT_ROOT}/video_scene_splitter.py"

echo "Output directory: ${OUTPUT_DIR}"

# Iterate over all mp4 files in the directory
for video_file in "$VIDEO_DIR"/*.mp4; do
    # Check whether the file exists to avoid errors when no mp4 files are present
    if [ -f "$video_file" ]; then
        echo "Processing: $video_file"
        python "$SCRIPT_PATH" --video_path "$video_file" --output_dir "$OUTPUT_DIR"
        echo "Finished: $video_file"
        echo "-----------------------------------"
    fi
done

echo "All videos have been processed!"
