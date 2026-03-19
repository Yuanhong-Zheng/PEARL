from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
import numpy as np
import torch
import os





# Define a list of query texts
queries = [
    {"text": "What is Xiao Ming sitting on while using his laptop?"}
]

documents = [
    {"text": "a man is running"},
    {"image": "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/annotations/negative.png"},
]

# Specify the model path
model_name_or_path = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/Qwen3-VL-Embedding-2B"

# Initialize the Qwen3VLEmbedder model
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)

# Combine queries and documents into a single input list
inputs = queries + documents

# Process the inputs to get embeddings
embeddings = model.process(inputs)
# import ipdb;ipdb.set_trace()

# Compute similarity scores between query embeddings and document embeddings
A=len(queries)
similarity_scores = (embeddings[:A] @ embeddings[A:].T)

# Print out the similarity scores in a list format
print(similarity_scores.tolist())

# [[0.8157786130905151, 0.7178360223770142, 0.7173429131507874], [0.5195091962814331, 0.3302568793296814, 0.4391537308692932], [0.3884059488773346, 0.285782128572464, 0.33141762018203735], [0.1092604324221611, 0.03871120512485504, 0.06952016055583954]]