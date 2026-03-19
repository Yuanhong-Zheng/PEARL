from flask import Flask, request, jsonify
import importlib
import sys
from pathlib import Path

qwen_repo_root = Path(__file__).resolve().parents[1] / "third_party" / "Qwen3-VL-Embedding"
if str(qwen_repo_root) not in sys.path:
    sys.path.insert(0, str(qwen_repo_root))
Qwen3VLEmbedder = importlib.import_module("src.models.qwen3_vl_embedding").Qwen3VLEmbedder

import numpy as np
import torch
import os

app = Flask(__name__)

# Specify the model path
model_name_or_path = "models/Qwen3-VL-Embedding-2B"

# Initialize the Qwen3VLEmbedder model (only once when the server starts)
print("正在加载模型...")
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
print("模型加载完成！")


@app.route('/compute_similarity', methods=['POST'])
def compute_similarity():
    """
    接收 queries 和 documents，返回相似度分数
    
    请求格式:
    {
        "queries": [
            {"text": "query text 1"},
            {"text": "query text 2"},
            {"image": "/path/to/image.png"},
            {"video": "/path/to/video.mp4"}
        ],
        "documents": [
            {"text": "document text 1"},
            {"image": "/path/to/image.png"},
            {"video": "/path/to/video.mp4"}
        ]
    }
    
    返回格式:
    {
        "similarity_scores": [[score1, score2, ...], [...]]
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求体不能为空"}), 400
        
        queries = data.get('queries', [])
        documents = data.get('documents', [])
        
        if not queries:
            return jsonify({"error": "queries 不能为空"}), 400
        
        if not documents:
            return jsonify({"error": "documents 不能为空"}), 400
        
        # Combine queries and documents into a single input list
        inputs = queries + documents
        
        # Process the inputs to get embeddings
        embeddings = model.process(inputs)
        
        # Compute similarity scores between query embeddings and document embeddings
        A = len(queries)
        similarity_scores = (embeddings[:A] @ embeddings[A:].T)
        
        # Return the similarity scores
        return jsonify({
            "similarity_scores": similarity_scores.tolist()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_embeddings', methods=['POST'])
def get_embeddings():
    """
    接收输入列表，返回对应的 embeddings
    
    请求格式:
    {
        "inputs": [
            {"text": "query text 1"},
            {"text": "query text 2"},
            {"image": "/path/to/image.png"},
            {"video": "/path/to/video.mp4"}
        ]
    }
    
    返回格式:
    {
        "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "请求体不能为空"}), 400
        
        inputs = data.get('inputs', [])
        
        if not inputs:
            return jsonify({"error": "inputs 不能为空"}), 400
        
        # Process the inputs to get embeddings
        embeddings = model.process(inputs)
        
        # Return the embeddings
        return jsonify({
            "embeddings": embeddings.tolist()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """健康检查端点"""
    return jsonify({"status": "healthy", "model_loaded": True})


@app.route('/', methods=['GET'])
def index():
    """服务信息"""
    return jsonify({
        "service": "Qwen3-VL-Embedding API",
        "endpoints": {
            "/compute_similarity": "POST - 计算查询和文档之间的相似度",
            "/get_embeddings": "POST - 获取输入的 embedding 向量",
            "/health": "GET - 健康检查",
            "/": "GET - 服务信息"
        },
        "example_requests": {
            "compute_similarity": {
                "url": "/compute_similarity",
                "method": "POST",
                "body": {
                    "queries": [
                        {"text": "What is Xiao Ming sitting on while using his laptop?"}
                    ],
                    "documents": [
                        {"text": "a man is running"},
                        {"image": "/path/to/image.png"},
                        {"video": "/path/to/video.mp4"}
                    ]
                }
            },
            "get_embeddings": {
                "url": "/get_embeddings",
                "method": "POST",
                "body": {
                    "inputs": [
                        {"text": "query text 1"},
                        {"text": "query text 2"},
                        {"image": "/path/to/image.png"},
                        {"video": "/path/to/video.mp4"}
                    ]
                }
            }
        }
    })


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-VL Embedding Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=5000, help="服务端口")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=False)