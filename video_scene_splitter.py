#!/usr/bin/env python3
"""
视频场景分割工具
使用 PySceneDetect 将视频按场景切分成多个 clip
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
    使用 ffmpeg 切割单个场景
    
    参数:
        video_path: 输入视频路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        output_path: 输出文件路径
        scene_idx: 场景索引（用于显示进度）
        total_scenes: 总场景数
    
    返回:
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
        '-y',  # 覆盖已存在的文件
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
    将视频按场景分割成多个 clip
    
    参数:
        video_path: 输入视频路径
        output_dir: 输出目录
        threshold: 场景检测阈值 (0-255)，值越小越敏感，默认 27.0
        min_scene_len: 最小场景长度（帧数），默认 15 帧
        min_clip_duration: 最小 clip 时长（秒），过滤掉小于该时长的片段，默认 3.0 秒
        max_clip_duration: 最大 clip 时长（秒），超过该时长的场景会被分割成多个片段，默认 10.0 秒
    """
    
    # 固定输出格式为 mp4，默认显示进度条
    output_format = "mp4"
    show_progress = True
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 在output_dir下创建以视频名字命名的子文件夹
    clips_output_dir = os.path.join(output_dir, video_name)
    
    # 如果该视频的输出目录已存在，跳过处理
    if os.path.exists(clips_output_dir):
        print(f"检测到已存在的输出目录，跳过处理: {clips_output_dir}")
        json_output_path = os.path.join(clips_output_dir, f"{video_name}_clips_info.json")
        if os.path.exists(json_output_path):
            print(f"✓ 已存在的 JSON 信息文件: {json_output_path}")
        return
    
    # 创建新的输出目录
    os.makedirs(clips_output_dir, exist_ok=True)
    
    # 创建视频管理器
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    
    # 添加内容检测器
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    
    # 设置基础时间码
    base_timecode = video_manager.get_base_timecode()
    
    # 开始视频管理器
    video_manager.set_downscale_factor()
    video_manager.start()
    
    print(f"开始分析视频: {video_path}")
    print(f"检测阈值: {threshold}")
    print(f"最小场景长度: {min_scene_len} 帧")
    
    # 执行场景检测
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=show_progress)
    
    # 获取检测到的场景列表
    scene_list = scene_manager.get_scene_list(base_timecode)
    
    print(f"\n检测到 {len(scene_list)} 个场景")
    
    if len(scene_list) == 0:
        print("未检测到场景切换，将按最大时长进行均匀切割")
        # 用 ffprobe 获取视频总时长
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
            print("无法获取视频总时长，无法进行切割")
            return

        clip_len = max_clip_duration if max_clip_duration > 0 else total_duration
        final_clips = []
        t = 0.0
        while t < total_duration:
            end = min(t + clip_len, total_duration)
            if end - t >= min_clip_duration:
                final_clips.append((t, end))
            t = end

        print(f"视频总时长: {total_duration:.2f}s，按 {clip_len} 秒切割为 {len(final_clips)} 个片段")

        if len(final_clips) == 0:
            print(f"所有片段均小于 {min_clip_duration} 秒，无有效片段")
            return
    else:
        # 过滤掉时长小于 min_clip_duration 的场景
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
                print(f"过滤掉 {filtered_count} 个时长小于 {min_clip_duration} 秒的片段")
            
            scene_list = filtered_scene_list
            
            if len(scene_list) == 0:
                print(f"所有场景都小于 {min_clip_duration} 秒，无有效片段")
                return
        
        print(f"有效场景数: {len(scene_list)}")
        
        # 将超过 max_clip_duration 的场景分割成多个片段
        if max_clip_duration > 0:
            split_scenes = []
            discarded_count = 0
            for scene in scene_list:
                start_time = scene[0].get_seconds()
                end_time = scene[1].get_seconds()
                duration = end_time - start_time
                
                if duration > max_clip_duration:
                    # 需要分割成多个片段
                    num_splits = int(duration / max_clip_duration) + (1 if duration % max_clip_duration > 0 else 0)
                    for j in range(num_splits):
                        sub_start = start_time + j * max_clip_duration
                        sub_end = min(start_time + (j + 1) * max_clip_duration, end_time)
                        sub_duration = sub_end - sub_start
                        
                        # 检查分割后的片段时长是否满足最小时长要求
                        if sub_duration >= min_clip_duration:
                            split_scenes.append((sub_start, sub_end))
                        else:
                            discarded_count += 1
                else:
                    split_scenes.append((start_time, end_time))
            
            split_count = len(split_scenes) - len(scene_list)
            if split_count > 0:
                print(f"将超过 {max_clip_duration} 秒的场景进一步分割")
            if discarded_count > 0:
                print(f"丢弃 {discarded_count} 个小于 {min_clip_duration} 秒的尾部片段")
            
            # 使用分割后的场景列表（转换为元组格式）
            final_clips = split_scenes
        else:
            # 如果不限制最大时长，使用原始场景列表
            final_clips = [(scene[0].get_seconds(), scene[1].get_seconds()) for scene in scene_list]
    
    print(f"最终片段数: {len(final_clips)}")
    
    # 打印场景信息
    print("\n片段时间戳:")
    for i, clip in enumerate(final_clips):
        if isinstance(clip, tuple):
            start_time, end_time = clip
        else:
            start_time = clip[0].get_seconds()
            end_time = clip[1].get_seconds()
        duration = end_time - start_time
        print(f"  片段 {i+1:03d}: {start_time:.2f}s -> {end_time:.2f}s (时长: {duration:.2f}s)")
    
    # 使用多线程并行切割视频（避免帧重叠）
    print(f"\n开始切割视频到目录: {clips_output_dir}")
    
    # 获取视频帧率
    fps = video_manager.get_framerate()
    if fps:
        print(f"视频帧率: {fps:.2f} fps")
    else:
        print("视频帧率: 未知（将不强制 CFR 输出）")
    
    # 设置帧偏移量，用于避免相邻 clip 之间的帧重叠
    # 对于第一个 clip 之外的所有 clip，起始时间会增加这个偏移量
    frame_offset = 0  # 4 帧偏移
    if fps and fps > 0:
        time_offset = frame_offset / fps  # 转换为秒
        print(f"时间偏移量: {frame_offset} 帧 ({time_offset:.3f} 秒)")
    else:
        time_offset = 0.0
        print("时间偏移量: 0 帧 (0.000 秒)")
    
    # 准备所有切割任务
    tasks = []
    for i, clip in enumerate(final_clips):
        if isinstance(clip, tuple):
            start_time, end_time = clip
        else:
            start_time = clip[0].get_seconds()
            end_time = clip[1].get_seconds()
        
        # 对于非第一个 clip，给起始时间添加偏移量以避免重叠帧
        if i > 0:
            start_time += time_offset
        
        # 生成输出文件名
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
    
    # 使用多线程并行执行切割任务
    # 线程数设置为 CPU 核心数的 2 倍（因为主要是 I/O 操作）
    max_workers = 8
    print(f"\n准备使用最多 {max_workers} 个线程进行切割...")
    failed_scenes = []
    
    print(f"使用 {max_workers} 个线程并行处理...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
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
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_task):
            scene_idx, success, error_msg = future.result()
            completed += 1
            
            if show_progress:
                task = future_to_task[future]
                print(f"  [{completed}/{len(final_clips)}] 片段 {scene_idx}: {task['start_time']:.2f}s - {task['end_time']:.2f}s {'✓' if success else '✗'}")
            
            if not success:
                failed_scenes.append((scene_idx, error_msg))
    
    # 显示失败的场景
    if failed_scenes:
        print(f"\n⚠️  警告: {len(failed_scenes)} 个片段切割失败:")
        for scene_idx, error_msg in failed_scenes:
            print(f"  片段 {scene_idx}: {error_msg[:100]}")
    
    print(f"\n✓ 完成！共生成 {len(final_clips) - len(failed_scenes)}/{len(final_clips)} 个视频片段")
    print(f"输出目录: {clips_output_dir}")
    
    # 生成 JSON 文件
    clips_info = []
    for i, clip in enumerate(final_clips):
        if isinstance(clip, tuple):
            start_time, end_time = clip
        else:
            start_time = clip[0].get_seconds()
            end_time = clip[1].get_seconds()
        
        # 应用时间偏移量（与切割时保持一致）
        if i > 0:
            start_time += time_offset
        
        # 生成 clip 文件名（与 split_video_ffmpeg 的命名规则一致）
        clip_filename = f"{video_name}_scene_{i+1:03d}.{output_format}"
        clip_path = os.path.join(clips_output_dir, clip_filename)
        
        # 转换为绝对路径
        clip_abs_path = os.path.abspath(clip_path)
        
        clips_info.append({
            "clip_id": i,
            "clip_path": clip_abs_path,
            "start_time": round(start_time, 2),
            "end_time": round(end_time, 2),
            "duration": round(end_time - start_time, 2),
            "concept1": False
        })
    
    # 创建完整的输出数据
    output_data = {
        "source_video": os.path.abspath(video_path),
        "concept1_path": "",
        "total_clips": len(clips_info),
        "clips": clips_info
    }
    
    # 保存 JSON 文件
    json_output_path = os.path.join(clips_output_dir, f"{video_name}_clips_info.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON 信息文件已保存: {json_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用 PySceneDetect 将视频按场景分割成多个 clip"
    )
    
    parser.add_argument(
        "--video_path",
        type=str,
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/video-level/videos/jianshen1.mp4",
        help="输入视频路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/output_clips",
        help="输出目录，默认: ./output_clips"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="场景检测阈值 (0-255)，值越小越敏感，默认: 27.0"
    )
    
    parser.add_argument(
        "--min_scene_len",
        type=int,
        default=15,
        help="最小场景长度（帧数），默认: 15"
    )
    
    parser.add_argument(
        "--min_clip_duration",
        type=float,
        default=1.0,
        help="最小 clip 时长（秒），过滤掉小于该时长的片段，默认: 3.0"
    )
    
    parser.add_argument(
        "--max_clip_duration",
        type=float,
        default=8.0,
        # default=10.0,
        help="最大 clip 时长（秒），超过该时长的场景会被分割成多个片段，默认: 10.0"
    )
    
    args = parser.parse_args()
    
    # 执行视频分割
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
