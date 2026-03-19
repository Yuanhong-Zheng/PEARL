# -*- coding: utf-8 -*-
"""
Flask service exposing an OpenAI-compatible API for LLaVA-OneVision via vLLM.
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

# Global variables for the model and processor
llm = None
processor = None
model_path = None
video_frames = 4  # Number of video frames to sample


# def read_video_pyav(container, indices):
#     """Decode video using PyAV."""
#     frames = []
#     start_index = indices[0]
#     end_index = indices[-1]
#     indices_set = set(indices)

#     video_stream = container.streams.video[0]
#     average_rate = float(video_stream.average_rate)
#     time_base = float(video_stream.time_base)

#     print(f"  [read_video_pyav] start_index={start_index}, end_index={end_index}, target_indices={list(indices)}")

#     # Seek directly to the time position for start_index to avoid decoding from frame 0
#     if start_index > 0 and average_rate > 0 and time_base > 0:
#         target_pts = int(start_index / average_rate / time_base)
#         container.seek(target_pts, stream=video_stream, backward=True, any_frame=False)
#     else:
#         container.seek(0)

#     # Decode frames and compute absolute frame indices via pts
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
    
#     print(f"  [read_video_pyav] final_sampled_indices={sampled_indices}, total_frames={len(sampled_indices)}")
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def read_video_pyav(container, indices):
    """Decode a video with PyAV."""
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    print(f"  [read_video_pyav] start_index={start_index}, end_index={end_index}, target_indices={list(indices)}, sampled_indices={[i for i in range(start_index, end_index + 1) if i in indices]}, total_frames={len(frames)}")
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def read_image(image_path):
    """Load an image."""
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def load_media_from_url(url, media_type='image', target_frames=4, video_start=None, video_end=None):
    """
    Load a media file from a URL.

    Args:
        url: File path, optionally prefixed with file://
        media_type: 'image' or 'video'
        target_frames: Number of frames to sample uniformly (default: 4)
        video_start: Video start time in seconds; None means from the beginning
        video_end: Video end time in seconds; None means until the end
    """
    # Strip the file:// prefix if present
    if url.startswith('file://'):
        file_path = url[7:]
    else:
        file_path = url
    
    if media_type == 'image':
        return read_image(file_path)
    elif media_type == 'video':
        container = av.open(file_path)
        video_stream = container.streams.video[0]
        
        # Get the original FPS
        original_fps = float(video_stream.average_rate)
        
        # Get the total frame count and duration
        total_frames = video_stream.frames
        if total_frames == 0:
            # Fall back to estimating frame count from duration
            duration = video_stream.duration
            time_base = video_stream.time_base
            total_frames = int(duration * time_base * original_fps)
        
        # Compute the total video duration in seconds
        video_duration = total_frames / original_fps
        
        # Resolve video_start and video_end
        start_time = 0.0 if video_start is None else max(0.0, video_start)
        end_time = video_duration if video_end is None else min(video_duration, video_end)
        
        # Ensure end_time > start_time
        if end_time <= start_time:
            end_time = start_time + 0.1  # Keep at least 0.1 seconds
        
        # Convert the time range to frame indices
        start_frame = int(start_time * original_fps)
        end_frame = int(end_time * original_fps)
        
        # Clamp frame indices to the valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        # Uniformly sample frames inside the requested range
        indices = np.linspace(start_frame, end_frame - 1, target_frames, dtype=int)
        
        if video_start is not None or video_end is not None:
            print(f"  Video sampling info: original_fps={original_fps:.2f}, target_frames={target_frames}, "
                  f"total_frames={total_frames}, time_range={start_time:.2f}s-{end_time:.2f}s, "
                  f"frame_range={start_frame}-{end_frame}, sampled_frames={len(indices)}")
        else:
            print(f"  Video sampling info: original_fps={original_fps:.2f}, target_frames={target_frames}, "
                  f"total_frames={total_frames}, sampled_frames={len(indices)}")
        
        video = read_video_pyav(container, indices)
        container.close()
        return video
    else:
        raise ValueError(f"Unsupported media type: {media_type}")


def prepare_inputs_for_llava_ov(messages, processor):
    """
    Prepare LLaVA-OneVision inputs while preserving media order manually.

    Args:
        messages: List of messages in OpenAI format
        processor: AutoProcessor instance (unused, kept for API compatibility)

    Returns:
        A dict containing prompt and multi_modal_data
    """
    images = []
    videos = []
    prompt_parts = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        
        # Start the block for this role
        if role == "user":
            prompt_parts.append("<|im_start|>user ")
        elif role == "assistant":
            prompt_parts.append("<|im_start|>assistant ")
        else:
            prompt_parts.append(f"<|im_start|>{role} ")
        
        # Process content
        if isinstance(content, str):
            # Plain-text message
            prompt_parts.append(content)
        else:
            # Handle multimodal content in order
            i = 0
            while i < len(content):
                item = content[i]
                item_type = item.get("type", "")
                
                if item_type == "text":
                    text = item.get("text", "")
                    prompt_parts.append(text)
                    i += 1
                
                elif item_type in ["image_url", "video_url"]:
                    # Collect consecutive media placeholders
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
                            # Read optional video_start and video_end
                            video_start = video_url_dict.get("video_start", None)
                            video_end = video_url_dict.get("video_end", None)
                            # Read optional nframes or use the global default
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
                    
                    # Join consecutive media placeholders with spaces, then add a newline
                    prompt_parts.append(" ".join(media_parts))
                    prompt_parts.append("\n")
                
                else:
                    i += 1
        
        # End the block for this role
        prompt_parts.append("<|im_end|>\n")
    
    # Add the assistant prefix for generation
    prompt_parts.append("<|im_start|>assistant\n")
    
    # Build the final prompt
    prompt = "".join(prompt_parts)
    
    # Build multi_modal_data
    mm_data = {}
    if images:
        mm_data['image'] = images if len(images) > 1 else images[0]
    if videos:
        mm_data['video'] = videos if len(videos) > 1 else videos[0]
    
    print(f"  Prepared inputs: {len(images)} image(s), {len(videos)} video(s)")
    print(f"  Prompt preview (first 200 chars): {prompt[:200]}")
    return {
        'prompt': prompt,
        'multi_modal_data': mm_data
    }


def initialize_model(checkpoint_path: str, tensor_parallel_size: int = 1, target_frames: int = 4):
    """Initialize the model and processor."""
    global llm, processor, model_path, video_frames
    
    if llm is not None:
        print("Model is already loaded, skipping initialization")
        return
    
    print(f"Loading model: {checkpoint_path}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"Video sampling frames: {target_frames}")
    
    processor = AutoProcessor.from_pretrained(checkpoint_path)
    print("✓ Processor loaded")
    
    llm = LLM(
        model=checkpoint_path,
        tensor_parallel_size=tensor_parallel_size,
        seed=0,
        gpu_memory_utilization=0.7
    )
    print("✓ LLM model loaded")
    
    model_path = checkpoint_path
    video_frames = target_frames


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    try:
        data = request.json
        
        # Read request arguments
        messages = data.get('messages', [])
        max_tokens = data.get('max_tokens', 1024)
        temperature = data.get('temperature', 0.0)
        top_p = data.get('top_p', 0.95)
        top_k = data.get('top_k', -1)
        
        # Prepare model inputs
        inputs = prepare_inputs_for_llava_ov(messages, processor)
        
        # Configure sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop_token_ids=[],
        )
        
        # Measure generate time only, excluding media decoding and preprocessing
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
    
    parser = argparse.ArgumentParser(description="LLaVA-OneVision vLLM Flask service")
    parser.add_argument(
        "--model_path",
        type=str,
        # default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/ReKV/model_zoo/llava-onevision-qwen2-7b-ov-hf",
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/llava-onevision-qwen2-0.5b-ov-hf",
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
    parser.add_argument(
        "--video_frames",
        type=int,
        default=4,
        help="Number of video frames to sample (default: 4, uniform sampling)"
    )
    
    args = parser.parse_args()
    
    # Initialize the model
    print("=" * 80)
    print("Initializing LLaVA-OneVision vLLM Flask service")
    print("=" * 80)
    initialize_model(args.model_path, args.tensor_parallel_size, args.video_frames)
    
    print("\n" + "=" * 80)
    print(f"Starting Flask service on {args.host}:{args.port}")
    print("=" * 80 + "\n")
    
    # Start the Flask service
    app.run(host=args.host, port=args.port, debug=False, threaded=False)


if __name__ == "__main__":
    main()
