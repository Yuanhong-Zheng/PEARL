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
print("Loading model...")
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
print("Model loaded successfully!")


@app.route('/compute_similarity', methods=['POST'])
def compute_similarity():
    """
    Accept queries and documents, and return similarity scores.

    Request format:
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
    
    Response format:
    {
        "similarity_scores": [[score1, score2, ...], [...]]
    }
    """
    try:
        # Get request payload
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request body cannot be empty"}), 400
        
        queries = data.get('queries', [])
        documents = data.get('documents', [])
        
        if not queries:
            return jsonify({"error": "queries cannot be empty"}), 400
        
        if not documents:
            return jsonify({"error": "documents cannot be empty"}), 400
        
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
    Accept a list of inputs and return the corresponding embeddings.

    Request format:
    {
        "inputs": [
            {"text": "query text 1"},
            {"text": "query text 2"},
            {"image": "/path/to/image.png"},
            {"video": "/path/to/video.mp4"}
        ]
    }
    
    Response format:
    {
        "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
    }
    """
    try:
        # Get request payload
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Request body cannot be empty"}), 400
        
        inputs = data.get('inputs', [])
        
        if not inputs:
            return jsonify({"error": "inputs cannot be empty"}), 400
        
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
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": True})


@app.route('/', methods=['GET'])
def index():
    """Service metadata."""
    return jsonify({
        "service": "Qwen3-VL-Embedding API",
        "endpoints": {
            "/compute_similarity": "POST - Compute similarity between queries and documents",
            "/get_embeddings": "POST - Get embedding vectors for the provided inputs",
            "/health": "GET - Health check",
            "/": "GET - Service metadata"
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
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=5000, help="Server port")
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=False)
