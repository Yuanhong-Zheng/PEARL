"""Work around PyTorch 2.9's low-precision Conv3D performance regression.

For Qwen's already separated, non-overlapping patches the convolution is exactly
a linear projection of each patch. This is the same algebra used by vLLM's
Conv3dLayer PyTorch-2.9 workaround (issues vllm#27406 and pytorch#166122).
The original Conv3D parameters and state-dict layout remain in place.
"""

import types

import torch
import torch.nn.functional as F


def linear_patch_forward(self, hidden_states):
    weight = self.proj.weight
    flattened = hidden_states.reshape(-1, weight[0].numel()).to(dtype=weight.dtype)
    return F.linear(flattened, weight.reshape(self.embed_dim, -1), self.proj.bias)


def enable_torch29_patch_embed_workaround(model):
    if torch.__version__.split("+")[0] != "2.9.0":
        return 0
    count = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3VLVisionPatchEmbed":
            continue
        projection = module.proj
        if projection.stride != projection.kernel_size or any(projection.padding) or projection.groups != 1 or any(x != 1 for x in projection.dilation):
            raise ValueError("Unexpected patch convolution; equivalent linear path is not applicable")
        module.forward = types.MethodType(linear_patch_forward, module)
        count += 1
    if not count:
        raise ValueError("Expected a Qwen3-VL patch projection for the Torch 2.9 workaround")
    return count
