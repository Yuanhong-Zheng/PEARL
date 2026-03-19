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

### 5. Prepare model checkpoints

By default, the code expects local checkpoints under [`models/`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/models):

```text
models/
  Qwen3-VL-8B-Instruct/
  Qwen3-VL-Embedding-2B/
```

These paths are used by:

- [`server/qwenvl_flask_server.py`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/server/qwenvl_flask_server.py)
- [`server/embedding_server.py`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/server/embedding_server.py)

If your checkpoints live elsewhere, update the default `model_path` arguments in those files or pass explicit paths at launch time.

### 6. Prepare workspace directories

The repository uses the following directories during runtime:

- [`data/`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/data) for datasets and annotations
- [`.cache/`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/.cache) for concept databases, extracted frames, and embedding caches
- [`output_results/`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/output_results) for evaluation outputs
- [`server/logs/`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/server/logs) for service logs

Create them if needed:

```bash
mkdir -p data models .cache output_results server/logs
```

### 7. Optional: start the local services

For multi-GPU serving, the repository already includes:

```bash
bash scripts/start_multi_gpu_servers.sh 8
```

This starts:

- an OpenAI-compatible vLLM server from [`server/qwenvl_flask_server.py`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/server/qwenvl_flask_server.py)
- an embedding service from [`server/embedding_server.py`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/server/embedding_server.py)

After the services are running, the main inference script [`video_qa_inference.py`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/video_qa_inference.py) can talk to them through the default local endpoints.
