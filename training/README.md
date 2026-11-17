# PEARL training

Code for constructing PSVU-10K and for training and evaluating the two trained PEARL
components: the query side of the retriever (Qwen3-VL-Embedding-2B, LoRA, multi-positive
contrastive loss with frozen clip embeddings) and the answering VLM (Qwen3-VL-8B-Instruct,
LoRA SFT over four option rotations). The inference pipeline itself is unchanged by
training.

The training code reads the released inference code at the repository root (answer
parser, query rewriting, Qwen3-VL-Embedding model file), so it must stay inside a clone of
this repository. Run the commands below from this `training/` directory unless noted
otherwise. `../` is the repository root, which is the `--release-root` of the scripts.

## Layout

```text
src/pearl_training/   library: data construction, verification, training, evaluation
scripts/              command-line entry points (listed below)
configs/              teacher-service configuration used for PSVU-10K construction
pyproject.toml        package definition
requirements-narrative-s3.txt   extra packages (boto3) for fetching the source videos from an S3-compatible store
```

The training hyperparameters are in `../configs/train_retriever.env` and
`../configs/train_reader.env`, and `../scripts/train_retriever.sh` and
`../scripts/train_reader.sh` launch the two runs with them.

## Setup

From the repository root, after the inference installation in the main README:

```bash
pip install -r requirements-train.txt          # peft
pip install -e training --no-deps              # the pearl_training package
pip install -r training/requirements-narrative-s3.txt   # only for fetching the source videos
```

`--no-deps` keeps the locally modified `third_party/qwen-vl-utils` in place.
`io.code_revision()` records `git rev-parse HEAD` in every run record, so use a git clone
of this repository.

Models: `Qwen3-VL-8B-Instruct` and `Qwen3-VL-Embedding-2B` (see the main README).
The PSVU-10K teacher (Qwen3.8-Flash-Next) is served with vLLM by `scripts/serve_flash_next.py`,
which waits until every file listed in `<state-dir>/modelscope_files.json` is present in
`--model` and then starts `vllm serve` from the given `--environment`.

## 1. PSVU-10K construction

This section documents how PSVU-10K was built. The released data is described in the
main README.

PSVU-10K is built from YouTube videos of narrative works, mostly movies. The raw videos
are not redistributed: users download the YouTube videos themselves and make them
available at the S3-compatible store set as `ENDPOINT` in
`src/pearl_training/narrative_sources.py` (a placeholder).
`prepare_narrative_sources.py download` copies them to local files, checking them with
`ffprobe`; all later steps read those local files.

Each stage is an explicit command; run it with `--help` for all options.

| Stage | Entry point | Notes |
| --- | --- | --- |
| Rank candidate works by metadata | `scripts/screen_narrative_metadata.py prepare / screen` | GPT-6 Luna through an OpenAI-compatible gateway (`--base-url`, `--key-file`) |
| Select works and fetch their source videos | `scripts/prepare_narrative_sources.py select / download / export-manifest` | reads the user-downloaded YouTube videos from `ENDPOINT` (see above) |
| Visual recurrence screening | `scripts/screen_narrative_visual.py prepare / run` | Qwen3.8-Flash-Next checks that the same characters recur |
| Decode streams, 64-second windows | `scripts/prepare_narrative_streams.py`, `scripts/prepare_legacy_movie_sources.py`, `python -m pearl_training.narrative_scale_sources` | needs `ffmpeg`/`ffprobe` |
| Teacher service | `scripts/serve_flash_next.py --model ... --state-dir ... --environment ...` | vLLM, 8 GPUs, port 28790 |
| Question generation and the four verification gates | `scripts/generate_movies_legacy_v4.py --manifest ... --output ... --approval configs/local_flash_next_narrative10k_256.json --run`, `scripts/scale_legacy_movie_data.py` | single node |
| Multi-node generation | `scripts/run_continuous_fleet_factory.py` (coordinator) and `scripts/fleet_teacher_worker.py` (one per teacher node) | same generation and verification functions; requires a generic shared-disk task scheduler (`engine.py` in `../shared-fleet`, i.e. `shared-fleet/` at the repository root) that is not part of this repository; `configs/local_flash_next_narrative10k_fleet256.json` |
| Freeze the dataset and clip embeddings | `scripts/embedding_service.py --model <Qwen3-VL-Embedding-2B> --cache <dir>`, then `scripts/freeze_training_data.py --samples ... --output <frozen> --expected-families 10000` | writes `samples.jsonl`, `document_embeddings.pt`, `manifest.json` |
| Retriever text queries | `scripts/build_text_query_cache.py --samples <frozen>/samples.jsonl --helper-model <Qwen3-VL-8B-Instruct> --release-root .. --output <frozen>/release_text_queries.json --execute --base-url http://127.0.0.1:22003/v1` | base-model concept descriptions and query rewrites, as at inference |
| Token-budget preflight | `scripts/preflight_training_inputs.py` | CPU only |

Generation prompts and verifiers live in `legacy_movie_generation.py`,
`replenishment.py`, `restored_v4_qualification.py` and `stream_audit.py`. Every training
sample keeps its evidence clips, the verified negative clips and the receipts of the
verification gates; `schema.py` and `stream_audit.require_repaired_sample` check them
before training.

## 2. Training

Both runs use 8 GPUs, an effective batch size of 8, AdamW (weight decay 0.01), gradient
clipping at 1.0, LoRA rank 16 (alpha 32, dropout 0.05), BF16 and seed 42.

Retriever (query side only; clip embeddings stay at the frozen base encoder):

```bash
accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --module pearl_training.train_retrieval \
    --samples <frozen>/samples.jsonl --model <Qwen3-VL-Embedding-2B> --output <run>/retrieval \
    --max-updates 2244 --accumulation 1 --lr 2e-5 --rank 16 --max-length 8192 --seed 42 \
    --expected-gpus 8 --temperature 0.07 \
    --document-cache <frozen>/document_embeddings.pt --query-cache <frozen>/release_text_queries.json
```

Answering VLM:

```bash
accelerate launch --multi_gpu --num_processes 8 --num_machines 1 --mixed_precision bf16 \
    --dynamo_backend no --module pearl_training.train_sft \
    --samples <frozen>/samples.jsonl --model <Qwen3-VL-8B-Instruct> --output <run>/reader \
    --max-updates 9140 --accumulation 1 --lr 1e-4 --rank 16 --max-length 8192 --seed 42 \
    --expected-gpus 8 --save-every 1000
```

The answering VLM is trained on the evidence clips of each question plus at most two
verified negative clips (the training context of verification gate iii), with all four
option rotations. Training uses the same question-answering prompt format as evaluation
(Supp. D.3, `build_messages` in `../video_qa_inference.py`); it is built by
`multimodal.sft_messages` (protocol `pearl-eval-v1`) as a single interleaved user message:

1. the reference frame of each concept, followed by its definition, e.g.
   `The man on the left wearing a dark jacket is named {Carmyn}. Please remember this name.`;
2. `Here are the previous related video clips:` followed by the historical clips;
3. `The following is the current video clip:` followed by the current clip;
4. the question, its options `A.`-`D.` on separate lines, a blank line and
   `Please output your answer choice in <ans></ans> tags.`

The target is `<ans>X</ans>`. Inputs longer than 8,192 tokens are rejected, not truncated.
Both runs evaluate the held-out PSVU-10K validation split before and after training
(`validation_before.json`, `validation_after.json`).

`python -m pearl_training.trusted_snapshot_entry --target-module <module> --trusted-snapshot <frozen> ...`
wraps either module to reuse the validation of an already checked frozen snapshot.

## 3. Evaluation on PSVU-Bench

The trained adapters are evaluated with the unchanged inference pipeline on the
frame-level entity videos of PSVU-Bench (`../data/frame-level/`):

```bash
python scripts/run_public112_comparison.py --training-run <run> --output <eval_dir> \
    --release-root .. --embedding-base <Qwen3-VL-Embedding-2B> \
    --reader-base <Qwen3-VL-8B-Instruct> --ffmpeg-dir <dir containing ffmpeg and ffprobe> --execute
python scripts/analyze_public112_comparison.py --output <eval_dir>
```

`<run>` contains the `retrieval/` and `reader/` adapters. The comparison runs four arms
(`baseline`, `embedding_only`, `reader_only`, `joint`) and starts its own model services
(`scripts/serve_public112_adapters.py`). Without `--execute` it only writes the plan.
`<eval_dir>` must lie outside the repository. Each arm writes the usual
`*_result.json` and `*_evaluation.json` files to `<eval_dir>/<arm>/results/`, which
`python -m eval.main_table` (run from the repository root) turns into the per-category
table. `scripts/evaluate_retriever_adapter.py` evaluates a query adapter on a fixed
validation set and frozen clip embeddings.

To evaluate released adapters without a local training run, start the same adapter
services with `../scripts/start_lora_servers.sh` and run the standard inference script
(see the main README).
