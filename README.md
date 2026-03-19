# PEARL

## Install

PEARL relies on a local multimodal LLM server, a local embedding server, and several third-party vision libraries. The project currently assumes a Linux environment with CUDA GPUs.

### 1. Create a Python environment

We recommend Python 3.11:

```bash
conda create -n pearl python=3.11 -y
conda activate pearl
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install ffmpeg

```bash
conda install -c conda-forge ffmpeg
```

### 4. Install local third-party packages

The repository vendors two local dependencies under `third_party/`.

```bash
pip install -e third_party/qwen-vl-utils --no-deps
pip install -e third_party/Qwen3-VL-Embedding --no-deps
```

This makes imports such as `qwen_vl_utils` and the local Qwen3-VL embedding code available to the PEARL servers.
