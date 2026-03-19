# -*- coding: utf-8 -*-
"""
Flask service exposing an OpenAI-compatible API via vLLM.
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

# Global variables for the model and processor
llm = None
processor = None
model_path = None


def prepare_inputs_for_vllm(messages, processor):
    """Prepare inputs for vLLM."""
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
    """Initialize the model and processor."""
    global llm, processor, model_path
    
    if llm is not None:
        print("Model is already loaded, skipping initialization")
        return
    
    print(f"Loading model: {checkpoint_path}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    print("✓ Processor loaded")
    
    llm = LLM(
        model=checkpoint_path,
        mm_encoder_tp_mode="data",
        tensor_parallel_size=tensor_parallel_size,
        seed=0,
        gpu_memory_utilization=0.8,
        max_model_len=200000,
    )
    print("✓ LLM model loaded")
    
    model_path = checkpoint_path


def convert_openai_messages_to_qwen_format(openai_messages):
    """
    Convert OpenAI-format messages to Qwen format.

    OpenAI format:
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file://..."}}, ...]}]

    Qwen format:
    [{"role": "user", "content": [{"type": "image", "image": "file://..."}, {"type": "video", "video": "file://..."}, ...]}]
    
    Args:
        openai_messages: List of messages in OpenAI format
    """
    qwen_messages = []
    
    for msg in openai_messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        
        if isinstance(content, str):
            # Plain-text message
            qwen_messages.append({"role": role, "content": content})
            continue
        
        # Handle multimodal content
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
                # Build the video item and keep optional parameters
                video_element = {"type": "video", "video": video_url}
                # Forward optional arguments such as nframes or video_start
                for key in ["nframes", "fps", "min_frames", "max_frames", "video_start", "video_end"]:
                    if key in video_url_dict:
                        video_element[key] = video_url_dict[key]
                qwen_content.append(video_element)
            
            else:
                # Unsupported content types are not expected here
                print("Error: unsupported video/content type")
                import sys;sys.exit(0)
        
        qwen_messages.append({"role": role, "content": qwen_content})
    
    return qwen_messages


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.json
        
        # Read request arguments
        openai_messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 1024)
        temperature = data.get('temperature', 0.0)
        top_k = data.get('top_k', -1)
        
        # Convert OpenAI-format messages to Qwen format
        qwen_messages = convert_openai_messages_to_qwen_format(openai_messages)
        
        # Prepare model inputs
        inputs = prepare_inputs_for_vllm(qwen_messages, processor)
        
        # Configure sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            stop_token_ids=[],
        )
        
        # Measure generate time only, excluding multimodal preprocessing
        generate_t0 = time.perf_counter()
        outputs = llm.generate([inputs], sampling_params=sampling_params)
        inference_time_ms = round((time.perf_counter() - generate_t0) * 1000)
        generated_text = outputs[0].outputs[0].text
        
        # Build an OpenAI-style response
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
            # Custom field: model-only inference time in milliseconds
            "inference_time_ms": inference_time_ms
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "model_loaded": llm is not None,
        "model_path": model_path
    })


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Flask service")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/Qwen3-VL-8B-Instruct",
        help="Model path"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=22003,
        help="Server port"
    )
    
    args = parser.parse_args()
    
    # Initialize the model
    print("=" * 80)
    print("Initializing vLLM Flask service")
    print("=" * 80)
    initialize_model(args.model_path, args.tensor_parallel_size)
    
    print("\n" + "=" * 80)
    print(f"Starting Flask service on {args.host}:{args.port}")
    print("=" * 80 + "\n")
    
    # Start the Flask service
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
