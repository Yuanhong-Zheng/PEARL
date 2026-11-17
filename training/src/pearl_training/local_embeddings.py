from __future__ import annotations

import numpy as np
import requests

from .multimodal import document_messages, query_messages


class LocalEmbeddings:
    def __init__(self, base="http://127.0.0.1:28780"):
        self.base = base
        self.session = requests.Session()
        self.session.trust_env = False
        self.documents = {}

    def encode(self, messages):
        response = self.session.post(self.base + "/encode", json={"messages": messages}, timeout=180)
        response.raise_for_status()
        return np.array(response.json()["vector"], dtype=np.float32)

    def document(self, clip):
        if clip["id"] not in self.documents:
            self.documents[clip["id"]] = self.encode(document_messages(clip))
        return self.documents[clip["id"]]

    def retrieve(self, sample, clips, k=4):
        if not clips:
            return []
        query = self.encode(query_messages(sample))
        matrix = np.stack([self.document(clip) for clip in clips])
        scores = matrix @ query
        return [{**clips[i], "retrieval_score": float(scores[i])} for i in np.argsort(-scores)[:k]]
