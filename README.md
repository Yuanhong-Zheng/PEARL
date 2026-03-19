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

Download the archive from one of the following mirrors:

- [Hugging Face](#)
- [ModelScope](https://www.modelscope.cn/datasets/YuanhongZheng/PEARL-Data)

After downloading, place the split archives under the project root, merge them, and then extract:

```bash
cat frame-level.tar.gz.part* > frame-level.tar.gz
tar -xzf frame-level.tar.gz
```

After extraction, the dataset should be available under `data/frame-level/`.

The extracted directory structure should include:

```text
data/
  frame-level/
    annotations/
    output_clips/
    videos/
```

## Models

Please download the following models and place them under `models/`:

- `Qwen3-VL-8B-Instruct`
- `Qwen3-VL-Embedding-2B`
- `llava-onevision-qwen2-7b-ov-hf`

## Evaluation

The evaluation pipeline has four stages:

1. Split each source video into scene clips.
2. Start the VLM server and the embedding server on each GPU.
3. Run multi-GPU inference over the annotation files.
4. Aggregate the final evaluation metrics.

### 1. Split Videos Into Scene Clips

Run the scene splitting script first:

```bash
bash scripts/split_scene.sh
```

This script scans `data/frame-level/videos/` for `.mp4` files and invokes `video_scene_splitter.py` to generate scene clips and clip metadata for each video.

For evaluation, the inference script expects:

- annotations under `data/frame-level/annotations/`
- scene clips under `data/frame-level/output_clips/`

### 2. Start The Model Servers

Launch the VLM server and the embedding server on all GPUs before inference:

```bash
bash scripts/start_multi_gpu_servers.sh 8
```

Replace `8` with the number of GPUs you want to use.

By default, the launcher uses:

- `models/Qwen3-VL-8B-Instruct` for `server/qwenvl_flask_server.py`
- `models/llava-onevision-qwen2-7b-ov-hf` for `server/llava_ov_flask_server.py`

### 3. Run Inference

After all servers are ready, run:

```bash
bash scripts/eval_qwen.sh 8
```

For each annotation file, the script produces:

- `*_result.json`: model predictions
- `*_evaluation.json`: per-file evaluation summary

### 4. Aggregate Final Metrics

Once inference is finished, compute the overall metrics with:

```bash
python eval.py <output_dir>
```

This command reads all `*_evaluation.json` files in the result directory and reports:

- avg accuracy
- current-time QA accuracy
- past-time QA accuracy

It also saves the aggregated statistics to `overall_statistics.json` in the same output directory.

## TODO

1. Video-Level Data and Evaluation
2. Live Streaming Demo
