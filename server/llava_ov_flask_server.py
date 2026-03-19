# -*- coding: utf-8 -*-
"""
Flask 服务：使用 vLLM 提供 LLaVA-OneVision 的 OpenAI 兼容 API 接口
"""
import os
import time
import av
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

app = Flask(__name__)

# 全局变量存储模型和处理器
llm = None
processor = None
model_path = None
video_frames = 4  # 视频采样帧数


# def read_video_pyav(container, indices):
#     """使用 PyAV 解码视频"""
#     frames = []
#     start_index = indices[0]
#     end_index = indices[-1]
#     indices_set = set(indices)

#     video_stream = container.streams.video[0]
#     average_rate = float(video_stream.average_rate)
#     time_base = float(video_stream.time_base)

#     print(f"  [read_video_pyav] start_index={start_index}, end_index={end_index}, 目标indices={list(indices)}")

#     # 直接 seek 到 start_index 对应的时间位置，避免从第0帧逐帧解码
#     if start_index > 0 and average_rate > 0 and time_base > 0:
#         target_pts = int(start_index / average_rate / time_base)
#         container.seek(target_pts, stream=video_stream, backward=True, any_frame=False)
#     else:
#         container.seek(0)

#     # 解码帧，通过 pts 计算绝对帧索引
#     sampled_indices = []
#     for frame in container.decode(video=0):
#         if frame.pts is None:
#             continue
#         frame_idx = int(round(frame.pts * time_base * average_rate))
#         if frame_idx > end_index:
#             break
#         if frame_idx in indices_set:
#             frames.append(frame)
#             sampled_indices.append(frame_idx)
    
#     print(f"  [read_video_pyav] 最终采样到的idx={sampled_indices}, 共{len(sampled_indices)}帧")
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def read_video_pyav(container, indices):
    """使用 PyAV 解码视频"""
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    print(f"  [read_video_pyav] start_index={start_index}, end_index={end_index}, 目标indices={list(indices)}, 实际采样到的idx={[i for i in range(start_index, end_index + 1) if i in indices]}, 共{len(frames)}帧")
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def read_image(image_path):
    """读取图片"""
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def load_media_from_url(url, media_type='image', target_frames=4, video_start=None, video_end=None):
    """
    从 URL 加载媒体文件
    
    Args:
        url: 文件路径（支持 file:// 前缀）
        media_type: 'image' 或 'video'
        target_frames: 目标采样帧数（默认：4帧，均匀采样）
        video_start: 视频起始时间（秒），None 表示从开头
        video_end: 视频结束时间（秒），None 表示到结尾
    """
    # 移除 file:// 前缀
    if url.startswith('file://'):
        file_path = url[7:]
    else:
        file_path = url
    
    if media_type == 'image':
        return read_image(file_path)
    elif media_type == 'video':
        container = av.open(file_path)
        video_stream = container.streams.video[0]
        
        # 获取视频的原始 fps
        original_fps = float(video_stream.average_rate)
        
        # 获取总帧数和视频时长
        total_frames = video_stream.frames
        if total_frames == 0:
            # 如果无法获取总帧数，通过 duration 计算
            duration = video_stream.duration
            time_base = video_stream.time_base
            total_frames = int(duration * time_base * original_fps)
        
        # 计算视频总时长（秒）
        video_duration = total_frames / original_fps
        
        # 处理 video_start 和 video_end 参数
        start_time = 0.0 if video_start is None else max(0.0, video_start)
        end_time = video_duration if video_end is None else min(video_duration, video_end)
        
        # 确保 end_time > start_time
        if end_time <= start_time:
            end_time = start_time + 0.1  # 至少保留 0.1 秒
        
        # 计算对应的帧索引范围
        start_frame = int(start_time * original_fps)
        end_frame = int(end_time * original_fps)
        
        # 确保帧索引在有效范围内
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        # 在指定范围内均匀采样
        indices = np.linspace(start_frame, end_frame - 1, target_frames, dtype=int)
        
        if video_start is not None or video_end is not None:
            print(f"  视频采样信息: 原始fps={original_fps:.2f}, 目标帧数={target_frames}, "
                  f"总帧数={total_frames}, 时间范围={start_time:.2f}s-{end_time:.2f}s, "
                  f"帧范围={start_frame}-{end_frame}, 采样帧数={len(indices)}")
        else:
            print(f"  视频采样信息: 原始fps={original_fps:.2f}, 目标帧数={target_frames}, "
                  f"总帧数={total_frames}, 采样帧数={len(indices)}")
        
        video = read_video_pyav(container, indices)
        container.close()
        return video
    else:
        raise ValueError(f"不支持的媒体类型: {media_type}")


def prepare_inputs_for_llava_ov(messages, processor):
    """
    准备 LLaVA-OneVision 输入，手动构造prompt以保持媒体顺序
    
    Args:
        messages: OpenAI 格式的消息列表
        processor: AutoProcessor 实例（未使用，保留接口兼容性）
    
    Returns:
        包含 prompt 和 multi_modal_data 的字典
    """
    images = []
    videos = []
    prompt_parts = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        
        # 开始该角色的消息
        if role == "user":
            prompt_parts.append("<|im_start|>user ")
        elif role == "assistant":
            prompt_parts.append("<|im_start|>assistant ")
        else:
            prompt_parts.append(f"<|im_start|>{role} ")
        
        # 处理内容
        if isinstance(content, str):
            # 纯文本消息
            prompt_parts.append(content)
        else:
            # 多模态内容，按顺序处理
            i = 0
            while i < len(content):
                item = content[i]
                item_type = item.get("type", "")
                
                if item_type == "text":
                    text = item.get("text", "")
                    prompt_parts.append(text)
                    i += 1
                
                elif item_type in ["image_url", "video_url"]:
                    # 收集连续的媒体标记
                    media_parts = []
                    while i < len(content) and content[i].get("type", "") in ["image_url", "video_url"]:
                        current_type = content[i].get("type", "")
                        
                        if current_type == "image_url":
                            image_url = content[i].get("image_url", {}).get("url", "")
                            image_data = load_media_from_url(image_url, media_type='image')
                            images.append(image_data)
                            media_parts.append("<image>")
                        
                        elif current_type == "video_url":
                            video_url_dict = content[i].get("video_url", {})
                            video_url = video_url_dict.get("url", "")
                            # 提取可选的 video_start 和 video_end 参数
                            video_start = video_url_dict.get("video_start", None)
                            video_end = video_url_dict.get("video_end", None)
                            # 提取可选的 nframes 参数，如果未指定则使用全局默认值
                            nframes = video_url_dict.get("nframes", video_frames)
                            video_data = load_media_from_url(
                                video_url, 
                                media_type='video', 
                                target_frames=nframes,
                                video_start=video_start,
                                video_end=video_end
                            )
                            videos.append(video_data)
                            media_parts.append("<video>")
                        
                        i += 1
                    
                    # 连续的媒体标记用空格连接，最后加一个换行
                    prompt_parts.append(" ".join(media_parts))
                    prompt_parts.append("\n")
                
                else:
                    i += 1
        
        # 结束该角色的消息
        prompt_parts.append("<|im_end|>\n")
    
    # 添加 assistant 的开始标记（用于生成）
    prompt_parts.append("<|im_start|>assistant\n")
    
    # 组合成最终的 prompt
    prompt = "".join(prompt_parts)
    
    # 构建 multi_modal_data
    mm_data = {}
    if images:
        mm_data['image'] = images if len(images) > 1 else images[0]
    if videos:
        mm_data['video'] = videos if len(videos) > 1 else videos[0]
    
    print(f"  准备输入: {len(images)} 个图片, {len(videos)} 个视频")
    print(f"  Prompt 预览 (前200字符): {prompt[:200]}")
    return {
        'prompt': prompt,
        'multi_modal_data': mm_data
    }


def initialize_model(checkpoint_path: str, tensor_parallel_size: int = 1, target_frames: int = 4):
    """初始化模型和处理器"""
    global llm, processor, model_path, video_frames
    
    if llm is not None:
        print("模型已经加载，跳过初始化")
        return
    
    print(f"正在加载模型: {checkpoint_path}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"视频采样帧数: {target_frames} 帧")
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    print("✓ Processor 加载完成")
    
    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=tensor_parallel_size,
        seed=0,
        gpu_memory_utilization=0.7
    )
    print("✓ LLM 模型加载完成")
    
    model_path = checkpoint_path
    video_frames = target_frames


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI 兼容的 chat completions 接口"""
    try:
        data = request.json
        
        # 获取参数
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 1024)
        temperature = data.get('temperature', 0.0)
        top_p = data.get('top_p', 0.95)
        top_k = data.get('top_k', -1)
        
        # 准备输入
        inputs = prepare_inputs_for_llava_ov(messages, processor)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_token_ids=[],
        )
        
        # 只统计模型 generate 阶段的耗时（不含视频/图像解码与预处理）
        generate_t0 = time.perf_counter()
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        inference_time_ms = round((time.perf_counter() - generate_t0) * 1000)
        generated_text = outputs[0].outputs[0].text
        
        # 构造 OpenAI 格式的响应
        response = {
            "id": "chatcmpl-" + os.urandom(12).hex(),
            "object": "chat.completion",
            "created": int(os.times().system),
            "model": model_path,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            },
            # 自定义字段：纯模型推理耗时（ms）
            "inference_time_ms": inference_time_ms
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "model_loaded": llm is not None,
        "model_path": model_path
    })


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLaVA-OneVision vLLM Flask 服务")
    parser.add_argument(
        "--model_path",
        type=str,
        # default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/ReKV/model_zoo/llava-onevision-qwen2-7b-ov-hf",
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/llava-onevision-qwen2-0.5b-ov-hf",
        help="模型路径"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="张量并行大小"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=22003,
        help="服务端口"
    )
    parser.add_argument(
        "--video_frames",
        type=int,
        default=4,
        help="视频采样帧数（默认：4帧，均匀采样）"
    )
    
    args = parser.parse_args()
    
    # 初始化模型
    print("=" * 80)
    print("初始化 LLaVA-OneVision vLLM Flask 服务")
    print("=" * 80)
    initialize_model(args.model_path, args.tensor_parallel_size, args.video_frames)
    
    print("\n" + "=" * 80)
    print(f"启动 Flask 服务: {args.host}:{args.port}")
    print("=" * 80 + "\n")
    
    # 启动 Flask 服务
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()

