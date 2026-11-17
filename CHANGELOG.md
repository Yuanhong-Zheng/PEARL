# Changelog

## v1.0 (2026-11)

Release for the updated paper *PEARL: Personalized Streaming Video Understanding with
Runtime-Defined Concepts* (arXiv 2603.20422).

### Added

- `training/`: PSVU-10K construction (source screening, question generation with four
  verification gates, dataset freezing), query-side LoRA training of the retriever
  (Qwen3-VL-Embedding-2B, multi-positive contrastive loss with frozen clip embeddings),
  LoRA training of the answering VLM (Qwen3-VL-8B-Instruct, four option rotations) and
  the evaluation of trained adapters. The answering VLM is trained with the evaluation
  prompt format (protocol `pearl-eval-v1`).
- `eval/main_table.py`: per-category PSVU-Bench table in the layout of the paper
  (Presence, Behavior, Appearance, Location, Relation, Action, Event, Time and the three
  averages) from the `*_evaluation.json` outputs and the new per-question `category`
  annotation field, with strict four-rotation scoring. Missing categories and other
  problems are reported. Unit tests in `tests/`.
- `configs/`: inference settings (K=4, N=1) and the LoRA training hyperparameters.
- `scripts/start_lora_servers.sh` to evaluate the trained adapters with the unchanged
  inference pipeline, and `scripts/train_retriever.sh` / `scripts/train_reader.sh`.
- `requirements-train.txt` (peft), `LICENSE` (MIT), `CHANGELOG.md`.
- README: overview of the task, PSVU-Bench, PEARL and PSVU-10K, the main results table
  with the trained model, data and model links, and training instructions.

### Changed

- The benchmark is renamed from PEARL-Bench to PSVU-Bench, and the README covers its
  Action split (`data/video-level/`) next to the frame-level split.
- `scripts/eval_qwen.sh` and `scripts/split_scene.sh` take the split (`SPLIT`) and the
  output folder from the environment, and `scripts/start_multi_gpu_servers.sh` takes the
  server script and model paths. The defaults are unchanged.
- `requirements.txt` pins `scenedetect<0.7`, since `video_scene_splitter.py` uses the
  `VideoManager` API removed in PySceneDetect 0.7.
- `third_party/Qwen3-VL-Embedding` is reduced to the unmodified model file used by the
  embedding server, with its license and a notice. `third_party/qwen-vl-utils` gains its
  license and a notice listing the local changes. The embedding package no longer needs
  `pip install -e`.
- License: code under the MIT License, annotations, metadata and synthetic videos under
  CC BY-NC 4.0, LoRA weights under Apache-2.0, replacing the license statement of v0.1.

### Removed

- Hard-coded absolute paths in script defaults, a leftover `ipdb` breakpoint in
  `video_qa_inference.py`, and the example, test and media files of the vendored
  Qwen3-VL-Embedding repository.

The inference pipeline and its default settings are unchanged.

## v0.1 (2026-03, tag `v0.1-arxiv-v1`)

Initial release with the first arXiv version (*PEARL: Personalized Streaming Video
Understanding Model*): PEARL inference code, the frame-level benchmark data (then called
PEARL-Bench) and the aggregate evaluation script `eval.py`.
