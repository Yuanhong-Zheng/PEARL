#!/usr/bin/env python3
"""
Video scene splitting tool.
Uses PySceneDetect to split a video into multiple clips by scene.
"""

import argparse
import os
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def split_single_scene(video_path, start_time, end_time, output_path, scene_idx, total_scenes, output_fps=None):
    """
    Split a single scene using ffmpeg.

    Args:
        video_path: Input video path
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output file path
        scene_idx: Scene index for progress display
        total_scenes: Total number of scenes

    Returns:
        (scene_idx, success, error_message)
    """
    duration = float(end_time) - float(start_time)
    if duration <= 0:
        return (scene_idx, False, f"Invalid clip duration: start={start_time}, end={end_time}")
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-vf', f"trim=start=0:end={duration},setpts=PTS-STARTPTS",
        '-af', f"atrim=start=0:end={duration},asetpts=PTS-STARTPTS",
    ]
    if output_fps is not None and output_fps > 0:
        cmd += ['-r', f"{output_fps:.3f}", '-fps_mode', 'cfr']
    cmd += [
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-pix_fmt', 'yuv420p',
        '-avoid_negative_ts', 'make_zero',
        '-movflags', '+faststart',
        '-y',  # Overwrite existing files
        output_path
    ]
    
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        return (scene_idx, False, result.stderr)
    else:
        return (scene_idx, True, None)


def split_video_by_scenes(
    video_path: str,
    output_dir: str = "./output_clips",
    threshold: float = 27.0,
    min_scene_len: int = 15,
    min_clip_duration: float = 3.0,
    max_clip_duration: float = 10.0
):
    """
    Split a video into multiple clips by detected scenes.

    Args:
        video_path: Input video path
        output_dir: Output directory
        threshold: Scene detection threshold (0-255); lower values are more sensitive
        min_scene_len: Minimum scene length in frames
        min_clip_duration: Minimum clip duration in seconds
        max_clip_duration: Maximum clip duration in seconds; longer scenes are split further
    """
    
    # Use mp4 output and show a progress bar by default
    output_format = "mp4"
    show_progress = True
    
    # Check whether the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    
    # Get the video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create a subdirectory named after the video inside output_dir
    clips_output_dir = os.path.join(output_dir, video_name)
    
    # Skip processing if the output directory for this video already exists
    if os.path.exists(clips_output_dir):
        print(f"Existing output directory detected, skipping: {clips_output_dir}")
        json_output_path = os.path.join(clips_output_dir, f"{video_name}_clips_info.json")
        if os.path.exists(json_output_path):
            print(f"✓ Existing JSON metadata file: {json_output_path}")
        return
    
    # Create a fresh output directory
    os.makedirs(clips_output_dir, exist_ok=True)
    
    # Create the video manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # Add the content detector
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    
    # Set the base timecode
    base_timecode = video_manager.get_base_timecode()
    
    # Start the video manager
    video_manager.set_downscale_factor()
    video_manager.start()
    
    print(f"Starting video analysis: {video_path}")
    print(f"Detection threshold: {threshold}")
    print(f"Minimum scene length: {min_scene_len} frame(s)")
    
    # Run scene detection
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=show_progress)
    
    # Get the detected scene list
    scene_list = scene_manager.get_scene_list(base_timecode)
    
    print(f"\nDetected {len(scene_list)} scene(s)")
    
    if len(scene_list) == 0:
        print("No scene changes detected; splitting uniformly by maximum clip duration")
        # Use ffprobe to get the total video duration
        probe_result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                video_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            total_duration = float(probe_result.stdout.strip())
        except ValueError:
            print("Unable to determine total video duration; cannot continue splitting")
            return

        clip_len = max_clip_duration if max_clip_duration > 0 else total_duration
        final_clips = []
        t = 0.0
        while t < total_duration:
            end = min(t + clip_len, total_duration)
            if end - t >= min_clip_duration:
                final_clips.append((t, end))
            t = end

        print(f"Total video duration: {total_duration:.2f}s, split into {len(final_clips)} segment(s) with clip length {clip_len} seconds")

        if len(final_clips) == 0:
            print(f"All segments are shorter than {min_clip_duration} seconds; no valid clips")
            return
    else:
        # Filter out scenes shorter than min_clip_duration
        if min_clip_duration > 0:
            filtered_scene_list = []
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                duration = end_time - start_time
                if duration >= min_clip_duration:
                    filtered_scene_list.append(scene)
            
            filtered_count = len(scene_list) - len(filtered_scene_list)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} clip(s) shorter than {min_clip_duration} seconds")
            
            scene_list = filtered_scene_list
            
            if len(scene_list) == 0:
                print(f"All scenes are shorter than {min_clip_duration} seconds; no valid clips")
                return
        
        print(f"Valid scenes: {len(scene_list)}")
        
        # Split scenes that exceed max_clip_duration into multiple clips
        if max_clip_duration > 0:
            split_scenes = []
            discarded_count = 0
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                duration = end_time - start_time
                
                if duration > max_clip_duration:
                    # Split into multiple sub-clips
                    num_splits = int(duration / max_clip_duration) + (1 if duration % max_clip_duration > 0 else 0)
                    for j in range(num_splits):
                        sub_start = start_time + j * max_clip_duration
                        sub_end = min(start_time + (j + 1) * max_clip_duration, end_time)
                        sub_duration = sub_end - sub_start
                        
                        # Check whether the sub-clip meets the minimum duration requirement
                        if sub_duration >= min_clip_duration:
                            split_scenes.append((sub_start, sub_end))
                        else:
                            discarded_count += 1
                else:
                    split_scenes.append((start_time, end_time))
            
            split_count = len(split_scenes) - len(scene_list)
            if split_count > 0:
                print(f"Further split scenes longer than {max_clip_duration} seconds")
            if discarded_count > 0:
                print(f"Discarded {discarded_count} trailing segment(s) shorter than {min_clip_duration} seconds")
            
            # Use the split scene list in tuple form
            final_clips = split_scenes
        else:
            # If no max duration is enforced, use the original scenes
            final_clips = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    
    print(f"Final clip count: {len(final_clips)}")
    
    # Print clip timestamps
    print("\nClip timestamps:")
    for i, clip in enumerate(final_clips):
        if isinstance(clip, tuple):
            start_time, end_time = clip
        else:
            start_time = clip[0].get_seconds()
            end_time = clip[1].get_seconds()
        duration = end_time - start_time
        print(f"  Clip {i+1:03d}: {start_time:.2f}s -> {end_time:.2f}s (duration: {duration:.2f}s)")
    
    # Split the video with multithreading
    print(f"\nStarting video splitting into directory: {clips_output_dir}")
    
    # Get the video frame rate
    fps = video_manager.get_framerate()
    if fps:
        print(f"Video frame rate: {fps:.2f} fps")
    else:
        print("Video frame rate: unknown (CFR output will not be forced)")
    
    # Set a frame offset to avoid overlapping frames between adjacent clips
    # For all clips except the first, the start time is shifted by this offset
    frame_offset = 0  # 4-frame offset
    if fps and fps > 0:
        time_offset = frame_offset / fps  # Convert to seconds
        print(f"Time offset: {frame_offset} frame(s) ({time_offset:.3f} seconds)")
    else:
        time_offset = 0.0
        print("Time offset: 0 frames (0.000 seconds)")
    
    # Prepare all split tasks
    tasks = []
    for i, clip in enumerate(final_clips):
        if isinstance(clip, tuple):
            start_time, end_time = clip
        else:
            start_time = clip[0].get_seconds()
            end_time = clip[1].get_seconds()
        
        # Shift the start time for non-first clips to avoid overlapping frames
        if i > 0:
            start_time += time_offset
        
        # Build the output filename
        output_filename = f"{video_name}_scene_{i+1:03d}.{output_format}"
        output_path = os.path.join(clips_output_dir, output_filename)
        
        tasks.append({
            'video_path': video_path,
            'start_time': start_time,
            'end_time': end_time,
            'output_path': output_path,
            'scene_idx': i + 1,
            'total_scenes': len(final_clips),
            'output_fps': fps
        })
    
    # Execute split tasks with multithreading
    # The thread count is tuned for I/O-heavy work
    max_workers = 8
    print(f"\nPreparing to split with up to {max_workers} threads...")
    failed_scenes = []
    
    print(f"Processing in parallel with {max_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {}
        for task in tasks:
            future = executor.submit(
                split_single_scene,
                task['video_path'],
                task['start_time'],
                task['end_time'],
                task['output_path'],
                task['scene_idx'],
                task['total_scenes'],
                task['output_fps']
            )
            future_to_task[future] = task
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_task):
            scene_idx, success, error_msg = future.result()
            completed += 1
            
            if show_progress:
                task = future_to_task[future]
                print(f"  [{completed}/{len(final_clips)}] Clip {scene_idx}: {task['start_time']:.2f}s - {task['end_time']:.2f}s {'✓' if success else '✗'}")
            
            if not success:
                failed_scenes.append((scene_idx, error_msg))
    
    # Show failed scenes
    if failed_scenes:
        print(f"\n⚠ Warning: failed to split {len(failed_scenes)} clip(s):")
        for scene_idx, error_msg in failed_scenes:
            print(f"  Clip {scene_idx}: {error_msg[:100]}")
    
    print(f"\n✓ Done! Generated {len(final_clips) - len(failed_scenes)}/{len(final_clips)} video clip(s)")
    print(f"Output directory: {clips_output_dir}")
    
    # Generate the JSON file
    clips_info = []
    for i, clip in enumerate(final_clips):
        if isinstance(clip, tuple):
            start_time, end_time = clip
        else:
            start_time = clip[0].get_seconds()
            end_time = clip[1].get_seconds()
        
        # Apply the same time offset used during splitting
        if i > 0:
            start_time += time_offset
        
        # Build the clip filename using the same naming convention
        clip_filename = f"{video_name}_scene_{i+1:03d}.{output_format}"
        clip_path = os.path.join(clips_output_dir, clip_filename)
        
        # Convert to an absolute path
        clip_abs_path = os.path.abspath(clip_path)
        
        clips_info.append({
            "clip_id": i,
            "clip_path": clip_abs_path,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(end_time - start_time, 2),
            "concept1": False
        })
    
    # Build the full output payload
    output_data = {
        "source_video": os.path.abspath(video_path),
        "concept1_path": "",
        "total_clips": len(clips_info),
        "clips": clips_info
    }
    
    # Save the JSON file
    json_output_path = os.path.join(clips_output_dir, f"{video_name}_clips_info.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON metadata file saved: {json_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Use PySceneDetect to split a video into multiple clips by scene"
    )
    
    parser.add_argument(
        "--video_path",
        type=str,
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/video-level/videos/jianshen1.mp4",
        help="Input video path"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/output_clips",
        help="Output directory, default: ./output_clips"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="Scene detection threshold (0-255); lower values are more sensitive, default: 27.0"
    )
    
    parser.add_argument(
        "--min_scene_len",
        type=int,
        default=15,
        help="Minimum scene length in frames, default: 15"
    )
    
    parser.add_argument(
        "--min_clip_duration",
        type=float,
        default=1.0,
        help="Minimum clip duration in seconds; shorter clips are filtered out, default: 3.0"
    )
    
    parser.add_argument(
        "--max_clip_duration",
        type=float,
        default=8.0,
        # default=10.0,
        help="Maximum clip duration in seconds; longer scenes are split into multiple clips, default: 10.0"
    )
    
    args = parser.parse_args()
    
    # Run scene-based splitting
    split_video_by_scenes(
        video_path=args.video_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        min_scene_len=args.min_scene_len,
        min_clip_duration=args.min_clip_duration,
        max_clip_duration=args.max_clip_duration
    )


if __name__ == "__main__":
    main()
