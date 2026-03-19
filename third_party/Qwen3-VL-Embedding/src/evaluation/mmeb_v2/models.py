from typing import Dict, Optional
import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoConfig

from ...models.qwen3_vl_embedding import Qwen3VLEmbedder


class MMEBEmbeddingModel(nn.Module):
    """Simplified MMEBModel for Qwen3VL embeddings."""

    def __init__(self,
                 encoder: Qwen3VLEmbedder,
                 normalize: bool = True,
                 temperature: float = 0.02):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

        # DDP setup
        self.is_ddp = dist.is_initialized()
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    @property
    def device(self):
        return self.encoder.model.device

    @property
    def config(self):
        return self.encoder.model.config

    @classmethod
    def load(cls,
             model_name_or_path: str,
             normalize: bool = True,
             temperature: float = 0.02,
             instruction: Optional[str] = None,
             **kwargs) -> "MMEBEmbeddingModel":
        """Load model from pretrained checkpoint."""
        default_instruction = kwargs.pop('default_instruction', instruction)
        encoder = Qwen3VLEmbedder(
            model_name_or_path=model_name_or_path,
            default_instruction=default_instruction or "Represent the user's input.",
            **kwargs
        )
        return cls(encoder=encoder, normalize=normalize, temperature=temperature)

    def save(self, output_dir: str):
        self.encoder.model.save_pretrained(output_dir)
        self.encoder.processor.save_pretrained(output_dir)

    def encode_input(self, inputs: Dict) -> Tensor:
        """Encode inputs using the Qwen3VL embedder.
        
        Args:
            inputs: Dict containing 'text', 'image', 'video', 'instruction' etc.
                    Can be a single dict or a list of dicts.
        """
        # 如果是预处理过的 tensor 输入，直接 forward
        if 'input_ids' in inputs:
            outputs = self.encoder.forward(inputs)
            hidden_state = outputs['last_hidden_state']
            attention_mask = outputs['attention_mask']
            pooled = self._pooling_last(hidden_state, attention_mask)
            if self.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            return pooled
        
        # 否则使用 embedder 的 process 方法
        if isinstance(inputs, dict):
            inputs = [inputs]
        return self.encoder.process(inputs, normalize=self.normalize)

    def _pooling_last(self, hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        """Pool the last non-padded token."""
        last_pos = attention_mask.flip(dims=[1]).argmax(dim=1)
        col = attention_mask.shape[1] - last_pos - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    def forward(self,
                qry: Dict[str, Tensor] = None,
                tgt: Dict[str, Tensor] = None) -> Dict:
        """Forward pass for contrastive learning / evaluation."""
        qry_reps = self.encode_input(qry) if qry else None
        tgt_reps = self.encode_input(tgt) if tgt else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        if self.is_ddp:
            all_qry = self._dist_gather(qry_reps)
            all_tgt = self._dist_gather(tgt_reps)
        else:
            all_qry, all_tgt = qry_reps, tgt_reps

        scores = torch.matmul(all_qry, all_tgt.T) / self.temperature
        target = torch.arange(scores.size(0), device=scores.device)
        target = target * (all_qry.size(0) // all_tgt.size(0))
        loss = self.cross_entropy(scores, target)

        if self.is_ddp:
            loss = loss * self.world_size
        return {"loss": loss, "qry_reps": qry_reps, "tgt_reps": tgt_reps}

    def _dist_gather(self, t: Tensor) -> Tensor:
        """Gather tensors across distributed processes."""
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        return torch.cat(all_tensors, dim=0)

    def compute_similarity(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        """Compute similarity matrix between query and passage representations."""
        return torch.matmul(q_reps, p_reps.T)


if __name__ == '__main__':
    model = MMEBEmbeddingModel.load(
        model_name_or_path=r'Your model path',
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16, device_map='cuda'
    )

    inputs = {
        'inputs': [
            {
                'text': "a woman breaks an egg",
                'instruction': 'Find images that corresponds to the given summary.',
            },
            {
                'text': "a woman breaks two eggs in a bowl",
                'instruction': 'Find images that corresponds to the given summary.',
            },
            {
                'image': r'https://ofasys-multimodal-wlcb-5-toshanghai.oss-cn-shanghai.aliyuncs.com/embedding_proj/linqi.lmx/codes/Qwen3-VL-Embedding/data/examples/0.jpeg?OSSAccessKeyId=LTAI5tSh9E5b7zDCb5uC8EsS&Expires=1925459675&Signature=fkEZFNVHMP3QaF49Qp2I%2Fz1GG6E%3D',
            },
            {
                'image': r'https://ofasys-multimodal-wlcb-5-toshanghai.oss-cn-shanghai.aliyuncs.com/embedding_proj/linqi.lmx/codes/Qwen3-VL-Embedding/data/examples/1.jpg?OSSAccessKeyId=LTAI5tSh9E5b7zDCb5uC8EsS&Expires=1925459699&Signature=PDrsu6gx5ivcskrxuISu2JUlRDc%3D',
            },
        ],
    }

    embeddings = model.encode_input(**inputs)

    print(
        f'Embeddings:\n{embeddings[:, :10].tolist()}\n{embeddings[:, -10:].tolist()}\n'
        f'Score:\n{model.compute_similarity(embeddings, embeddings).tolist()}\n'
    )
