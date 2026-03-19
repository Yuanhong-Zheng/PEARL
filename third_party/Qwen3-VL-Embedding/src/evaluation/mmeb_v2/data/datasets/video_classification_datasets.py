import os
import argparse
import shutil
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from .video_classification_utils import VIDEOCLS_LABEL_MAPPING, DATASET_INSTRUCTION
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...constant import EVAL_DATASET_HF_PATH
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import save_frames, process_video_frames


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    num_frames = kwargs['num_frames']
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    dataset_name = kwargs['dataset_name']

    # Get task instruction and normalize punctuation
    instruction = DATASET_INSTRUCTION.get(dataset_name, "Classify the video into the correct category.")
    if instruction.endswith(":"):
        instruction = instruction[:-1] + "."

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for video_id, label in zip(batch_dict['video_id'], batch_dict['pos_text']):
        # Process video and extract frames
        video_path = os.path.join(video_root, video_id + '.mp4')
        frame_dir = os.path.join(frame_root, video_id)
        if not os.path.exists(frame_dir):
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query input (video + instruction)
        query_inputs.append({
            "video": video_frame_paths,
            "instruction": instruction,
        })

        # Candidate input (positive label for current video)
        cand_inputs.append([{"text": label}])
        
        dataset_infos.append({
            "cand_names": [label],
            "label_name": label,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "video_classification"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_video_class_dataset(model_args, data_args, **kwargs):
    dataset_name = kwargs['dataset_name']
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[dataset_name])
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
    
    # Construct corpus: all class labels for this dataset
    corpus_list = []
    for label in VIDEOCLS_LABEL_MAPPING[dataset_name]:
        corpus_list.append({
            "cand_input": [{"text": label}],
            "dataset_infos": {"cand_names": [label]}
        })
    corpus = Dataset.from_list(corpus_list)

    return dataset, corpus


def export_videocls_dataset_for_hf(
    output_dir: str,
    dataset_name: str,
    max_queries: int = None,
    max_corpus: int = None,
    video_root: str = None,
    frame_root: str = None,
    max_frames_saved: int = 32,
    num_frames: int = 8,
):
    """
    Export video classification dataset in a format loadable by datasets.load_dataset
    
    Args:
        output_dir: Save directory
        dataset_name: Dataset name
        max_queries: Maximum number of queries
        max_corpus: Maximum number of corpus items (for classification, corpus is a fixed set of classes)
        video_root: Video file root directory
        frame_root: Frame extraction save directory
        max_frames_saved: Maximum frames to save
        num_frames: Number of frames for queries
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create video save directory
    videos_dir = output_path / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # Load original dataset
    print(f"Loading dataset: {dataset_name}")
    kwargs = {
        'dataset_name': dataset_name,
        'video_root': video_root or './videos',
        'frame_root': frame_root or './frames',
        'max_frames_saved': max_frames_saved,
        'num_frames': num_frames,
    }
    
    # Sampling parameters
    if max_queries:
        kwargs['sample_size'] = max_queries
    
    dataset, corpus = load_video_class_dataset(None, None, **kwargs)
    
    print(f"Dataset loaded with {len(dataset)} queries and {len(corpus)} classes")
    
    # Prepare three subsets
    queries_data = []
    corpus_data = []
    relevant_docs_data = []
    
    # Get task instruction
    instruction = DATASET_INSTRUCTION.get(dataset_name, "Classify the video into the correct category.")
    if instruction.endswith(":"):
        instruction = instruction[:-1] + "."
    
    # Build corpus (all class labels)
    label_to_corpus_id = {}
    for idx, corpus_item in enumerate(corpus):
        label = corpus_item['dataset_infos']['cand_names'][0]
        corpus_id = f"c_{idx}"
        label_to_corpus_id[label] = corpus_id
        
        corpus_data.append({
            "corpus_id": corpus_id,
            "text": label,
        })
        
        # If max_corpus limit is set
        if max_corpus and len(corpus_data) >= max_corpus:
            break
    
    print("Processing queries...")
    for idx, sample in enumerate(tqdm(dataset)):
        query_input = sample['query_input']
        dataset_info = sample['dataset_infos']
        
        query_id = f"q_{idx}"
        
        # Process video frames
        video_frames = query_input['video']
        saved_video_paths = []
        
        # Extract video_id from frame paths
        if video_frames:
            # Assume frame path format: frame_root/video_id/xxxx.jpeg
            video_id = Path(video_frames[0]).parent.name
        else:
            video_id = f"video_{idx}"
        
        for frame_idx, frame_path in enumerate(video_frames):
            if os.path.exists(frame_path):
                target_frame_path = videos_dir / f"{video_id}_{frame_idx:04d}.jpeg"
                if not target_frame_path.exists():
                    shutil.copy2(frame_path, target_frame_path)
                saved_video_paths.append(str(target_frame_path.relative_to(output_path)))
        
        # Build queries
        queries_data.append({
            "query_id": query_id,
            "instruction": query_input['instruction'],
            "video_paths": saved_video_paths,
            "video_id": video_id,
        })
        
        # Build relevant_docs (correct class)
        label = dataset_info['label_name']
        if label in label_to_corpus_id:
            corpus_id = label_to_corpus_id[label]
            relevant_docs_data.append({
                "query_id": query_id,
                "corpus_ids": [corpus_id],
            })
        
        # If max_queries limit is reached
        if max_queries and len(queries_data) >= max_queries:
            break
    
    # Convert to Dataset
    print("Creating HuggingFace datasets...")
    queries_dataset = Dataset.from_list(queries_data)
    corpus_dataset = Dataset.from_list(corpus_data)
    relevant_docs_dataset = Dataset.from_list(relevant_docs_data)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        "queries": queries_dataset,
        "corpus": corpus_dataset,
        "relevant_docs": relevant_docs_dataset,
    })
    
    # Save dataset
    print(f"Saving dataset to {output_dir}")
    dataset_dict.save_to_disk(output_dir)
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Export Summary")
    print("="*50)
    print(f"Dataset name: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"Queries: {len(queries_dataset)}")
    print(f"Corpus (classes): {len(corpus_dataset)}")
    print(f"Relevant docs: {len(relevant_docs_dataset)}")
    print(f"Videos directory: {videos_dir}")
    print(f"Instruction: {instruction}")
    print("="*50)
    
    # Save loading instructions
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# Video Classification Dataset Export: {dataset_name}

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

- **queries**: {len(queries_dataset)} samples
  - query_id: unique query identifier
  - instruction: classification task instruction
  - video_paths: list of video frame paths (relative to dataset root)
  - video_id: video identifier

- **corpus**: {len(corpus_dataset)} samples (class labels)
  - corpus_id: unique corpus identifier
  - text: class label text

- **relevant_docs**: {len(relevant_docs_dataset)} samples
  - query_id: corresponding query ID
  - corpus_ids: list containing the correct class corpus ID

## Task Description

**Instruction**: {instruction}

This is a video classification task where each query (video) should be matched to exactly one class label from the corpus.

## Videos

Video frames are stored in: `{videos_dir.relative_to(output_path)}/`

Each video is represented by {num_frames} frames extracted from up to {max_frames_saved} saved frames.

## Class Labels

Total number of classes: {len(corpus_dataset)}

Classes: {', '.join([item['text'] for item in corpus_data[:10]])}{'...' if len(corpus_data) > 10 else ''}
""")
    
    print(f"\nREADME saved to: {readme_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Video Classification dataset for HuggingFace datasets"
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
        required=True,
        help="Name of the dataset to load (must be in EVAL_DATASET_HF_PATH)"
    )
    
    parser.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Maximum number of queries to export (default: all)"
    )
    
    parser.add_argument(
        "--max_corpus",
        type=int,
        default=None,
        help="Maximum number of corpus items (classes) to export (default: all)"
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
        help="Number of frames to use for queries (default: 8)"
    )
    
    args = parser.parse_args()
    
    export_videocls_dataset_for_hf(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_queries=args.max_queries,
        max_corpus=args.max_corpus,
        video_root=args.video_root,
        frame_root=args.frame_root,
        max_frames_saved=args.max_frames_saved,
        num_frames=args.num_frames,
    )