# -*- coding: utf-8 -*-
"""
Flask 服务：使用 vLLM 提供 OpenAI 兼容的 API 接口
"""
import os
import time
import torch
from flask import Flask, request, jsonify
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

app = Flask(__name__)

# 全局变量存储模型和处理器
llm = None
processor = None
model_path = None


def prepare_inputs_for_vllm(messages, processor):
    """准备 vLLM 输入"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def initialize_model(checkpoint_path: str, tensor_parallel_size: int = 1):
    """初始化模型和处理器"""
    global llm, processor, model_path
    
    if llm is not None:
        print("模型已经加载，跳过初始化")
        return
    
    print(f"正在加载模型: {checkpoint_path}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    print("✓ Processor 加载完成")
    
    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=tensor_parallel_size,
        seed=0,
        gpu_memory_utilization=0.8,
        max_model_len=200000,
    )
    print("✓ LLM 模型加载完成")
    
    model_path = checkpoint_path


def convert_openai_messages_to_qwen_format(openai_messages):
    """
    将 OpenAI 格式的消息转换为 Qwen 格式
    
    OpenAI 格式:
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file://..."}}, ...]}]
    
    Qwen 格式:
    [{"role": "user", "content": [{"type": "image", "image": "file://..."}, {"type": "video", "video": "file://..."}, ...]}]
    
    Args:
        openai_messages: OpenAI 格式的消息列表
    """
    qwen_messages = []
    
    for msg in openai_messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        
        if isinstance(content, str):
            # 纯文本消息
            qwen_messages.append({"role": role, "content": content})
            continue
        
        # 处理多模态内容
        qwen_content = []
        for item in content:
            item_type = item.get("type", "")
            
            if item_type == "text":
                qwen_content.append({"type": "text", "text": item.get("text", "")})
            
            elif item_type == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                qwen_content.append({"type": "image", "image": image_url})
            
            elif item_type == "video_url":
                video_url_dict = item.get("video_url", {})
                video_url = video_url_dict.get("url", "")
                # 构建 video 元素，保留其他参数（如 nframes, video_start, video_end 等）
                video_element = {"type": "video", "video": video_url}
                # 传递其他参数（如 nframes, fps, min_frames, max_frames, video_start, video_end 等）
                for key in ["nframes", "fps", "min_frames", "max_frames", "video_start", "video_end"]:
                    if key in video_url_dict:
                        video_element[key] = video_url_dict[key]
                qwen_content.append(video_element)
            
            else:
                # 其他类型（如 video_segment）直接保持原样传递
                print("错误: 不支持的视频类型")
                import sys;sys.exit(0)
        
        qwen_messages.append({"role": role, "content": qwen_content})
    
    return qwen_messages


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI 兼容的 chat completions 接口"""
    try:
        data = request.json
        
        # 获取参数
        openai_messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 1024)
        temperature = data.get('temperature', 0.0)
        top_k = data.get('top_k', -1)
        
        # 将 OpenAI 格式转换为 Qwen 格式
        qwen_messages = convert_openai_messages_to_qwen_format(openai_messages)
        
        # 准备输入
        inputs = prepare_inputs_for_vllm(qwen_messages, processor)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            stop_token_ids=[],
        )
        
        # 只统计模型 generate 阶段的耗时（不含多模态预处理）
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
    
    parser = argparse.ArgumentParser(description="vLLM Flask 服务")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen3-VL-8B-Instruct",
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
    
    args = parser.parse_args()
    
    # 初始化模型
    print("=" * 80)
    print("初始化 vLLM Flask 服务")
    print("=" * 80)
    initialize_model(args.model_path, args.tensor_parallel_size)
    
    print("\n" + "=" * 80)
    print(f"启动 Flask 服务: {args.host}:{args.port}")
    print("=" * 80 + "\n")
    
    # 启动 Flask 服务
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()

