import os
import argparse
import shutil
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from ...constant import EVAL_DATASET_HF_PATH
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import save_frames, process_video_frames


TASK_INST_QRY = "Find a video that contains the following visual content."
# TASK_INST_TGT = "Understand the content of the provided video."


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    num_frames = kwargs['num_frames']
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for video_name, video_path, caption in zip(
        batch_dict['video_id'],
        batch_dict['video'],
        batch_dict['caption']
    ):
        query_inputs.append({
            "text": caption,
            "instruction": TASK_INST_QRY,
        })

        video_path = os.path.join(video_root, video_path)
        frame_dir = os.path.join(frame_root, video_name)
        if not os.path.exists(frame_dir):
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        cand_inputs.append([{
            "video": video_frame_paths,
            # "instruction": TASK_INST_TGT,
        }])
        
        dataset_infos.append({
            "cand_names": [video_name],
            "label_name": video_name,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "msrvtt"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_msrvtt_dataset(model_args, data_args, **kwargs):
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=256,
        num_proc=4,
        drop_last_batch=False,
        load_from_cache_file=False
    )
    dataset = dataset.select_columns(["query_input", "cand_input", "dataset_infos"])
    corpus = None

    return dataset, corpus


def export_msrvtt_dataset_for_hf(
    output_dir: str,
    dataset_name: str = "msrvtt",
    max_queries: int = None,
    max_corpus: int = None,
    video_root: str = None,
    frame_root: str = None,
    max_frames_saved: int = 32,
    num_frames: int = 8,
):
    """
    Export MSRVTT text-to-video retrieval dataset in a format loadable by datasets.load_dataset
    
    Args:
        output_dir: Output directory
        dataset_name: Dataset name
        max_queries: Limit number of queries
        max_corpus: Limit number of corpus items (videos)
        video_root: Root directory of video files
        frame_root: Directory for extracted frames
        max_frames_saved: Maximum number of frames to save
        num_frames: Number of frames to use for queries
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos_dir = output_path / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    print(f"Loading dataset: {dataset_name}")
    kwargs = {
        'dataset_name': dataset_name,
        'video_root': video_root or './videos',
        'frame_root': frame_root or './frames',
        'max_frames_saved': max_frames_saved,
        'num_frames': num_frames,
    }
    
    if max_queries:
        kwargs['sample_size'] = max_queries
    
    dataset, _ = load_msrvtt_dataset(None, None, **kwargs)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    queries_data = []
    corpus_data = {}  # Deduplicate by video_id
    relevant_docs_data = []
    
    print("Processing dataset...")
    for idx, sample in enumerate(tqdm(dataset)):
        query_input = sample['query_input']
        cand_input = sample['cand_input'][0]
        dataset_info = sample['dataset_infos']
        
        video_id = dataset_info['label_name']
        query_id = f"q_{idx}"
        corpus_id = f"c_{video_id}"
        
        video_frames = cand_input['video']
        saved_video_paths = []
        
        for frame_idx, frame_path in enumerate(video_frames):
            if os.path.exists(frame_path):
                target_frame_path = videos_dir / f"{video_id}_{frame_idx:04d}.jpeg"
                if not target_frame_path.exists():
                    shutil.copy2(frame_path, target_frame_path)
                saved_video_paths.append(str(target_frame_path.relative_to(output_path)))
        
        queries_data.append({
            "query_id": query_id,
            "text": query_input['text'],
            "instruction": query_input['instruction'],
        })
        
        if video_id not in corpus_data:
            corpus_data[video_id] = {
                "corpus_id": corpus_id,
                "video_id": video_id,
                "video_paths": saved_video_paths,
            }
        
        relevant_docs_data.append({
            "query_id": query_id,
            "corpus_ids": [corpus_id],
        })
        
        if max_corpus and len(corpus_data) >= max_corpus:
            break
    
    print("Creating HuggingFace datasets...")
    queries_dataset = Dataset.from_list(queries_data)
    corpus_dataset = Dataset.from_list(list(corpus_data.values()))
    relevant_docs_dataset = Dataset.from_list(relevant_docs_data)
    
    dataset_dict = DatasetDict({
        "queries": queries_dataset,
        "corpus": corpus_dataset,
        "relevant_docs": relevant_docs_dataset,
    })
    
    print(f"Saving dataset to {output_dir}")
    dataset_dict.save_to_disk(output_dir)
    
    print("\n" + "="*50)
    print("Dataset Export Summary")
    print("="*50)
    print(f"Dataset name: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"Queries (captions): {len(queries_dataset)}")
    print(f"Corpus (videos): {len(corpus_dataset)}")
    print(f"Relevant docs: {len(relevant_docs_dataset)}")
    print(f"Videos directory: {videos_dir}")
    print(f"Task: Text-to-Video Retrieval")
    print("="*50)
    
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# MSRVTT Text-to-Video Retrieval Dataset Export

## Loading the dataset

```python
from datasets import load_from_disk

dataset = load_from_disk("{output_dir}")

# Access subsets
queries = dataset["queries"]
corpus = dataset["corpus"]
relevant_docs = dataset["relevant_docs"]
```

## Dataset Structure

- **queries**: {len(queries_dataset)} samples (text captions)
  - query_id: unique query identifier
  - text: video caption/description
  - instruction: retrieval task instruction ("{TASK_INST_QRY}")

- **corpus**: {len(corpus_dataset)} samples (videos)
  - corpus_id: unique corpus identifier (c_{{video_id}})
  - video_id: original video identifier
  - video_paths: list of video frame paths (relative to dataset root)

- **relevant_docs**: {len(relevant_docs_dataset)} samples
  - query_id: corresponding query ID
  - corpus_ids: list containing the relevant video corpus ID

## Task Description

**Task Type**: Text-to-Video Retrieval

**Instruction**: {TASK_INST_QRY}

This is a text-to-video retrieval task where text queries (captions) are matched to relevant videos from the corpus.

## Videos

Video frames are stored in: `{videos_dir.relative_to(output_path)}/`

Each video is represented by {num_frames} frames extracted from up to {max_frames_saved} saved frames.

## Statistics

- Total unique videos: {len(corpus_dataset)}
- Total text queries: {len(queries_dataset)}
- Queries per video: {len(queries_dataset) / len(corpus_dataset):.2f} (average)

## Example Query

```python
query = queries[0]
print(f"Query ID: {{query['query_id']}}")
print(f"Text: {{query['text']}}")
print(f"Instruction: {{query['instruction']}}")

# Find relevant video
relevant = relevant_docs[0]
corpus_id = relevant['corpus_ids'][0]
video = corpus.filter(lambda x: x['corpus_id'] == corpus_id)[0]
print(f"Relevant video: {{video['video_id']}}")
print(f"Video frames: {{len(video['video_paths'])}} frames")
```
""")
    
    print(f"\nREADME saved to: {readme_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MSRVTT Text-to-Video Retrieval dataset for HuggingFace datasets"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the exported dataset"
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="msrvtt",
        help="Name of the dataset to load (default: msrvtt)"
    )
    
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Maximum number of queries (captions) to export (default: all)"
    )
    
    parser.add_argument(
        "--max_corpus",
        type=int,
        default=None,
        help="Maximum number of corpus items (videos) to export (default: all)"
    )
    
    parser.add_argument(
        "--video_root",
        type=str,
        default=None,
        help="Root directory containing video files (default: ./videos)"
    )
    
    parser.add_argument(
        "--frame_root",
        type=str,
        default=None,
        help="Root directory for extracted frames (default: ./frames)"
    )
    
    parser.add_argument(
        "--max_frames_saved",
        type=int,
        default=64,
        help="Maximum number of frames to save per video (default: 32)"
    )
    
    parser.add_argument(
        "--num_frames",
        type=int,
        default=64,
        help="Number of frames to use for each video (default: 8)"
    )
    
    args = parser.parse_args()
    
    export_msrvtt_dataset_for_hf(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_queries=args.max_queries,
        max_corpus=args.max_corpus,
        video_root=args.video_root,
        frame_root=args.frame_root,
        max_frames_saved=args.max_frames_saved,
        num_frames=args.num_frames,
    )