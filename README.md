# PEARL

## Install

### 1. Create a Python environment

We recommend Python 3.11:

```bash
conda create -n pearl python=3.11 -y
conda activate pearl
```

### 2. Install dependencies

Install the Python dependencies first:

```bash
pip install -r requirements.txt
```

Install `ffmpeg` with conda:

```bash
conda install -c conda-forge ffmpeg
```

Install the local third-party packages without pulling extra dependencies:

```bash
pip install -e third_party/qwen-vl-utils --no-deps
pip install -e third_party/Qwen3-VL-Embedding --no-deps
```

## Dataset

We provide the `frame-level` dataset as a single `.tar.gz` archive. Download the archive from one of the following mirrors:

- Hugging Face:
- ModelScope:

After downloading, place the archive under the project root and extract it:

```bash
tar -xzf frame-level.tar.gz -C data
```

After extraction, the dataset should be available under [`data/frame-level/`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/data/frame-level).

The extracted directory structure should include:

```text
data/
  frame-level/
    annotations_filtered/
    output_clips/
    videos/
```

## Models

Please download the following models and place them under [`models/`](/mdr5/user/quantaalpha/jiangtianyi/PEARL/models):

- `Qwen3-VL-8B-Instruct`
- `Qwen3-VL-Embedding-2B`
- `llava-onevision-qwen2-7b-ov-hf`
