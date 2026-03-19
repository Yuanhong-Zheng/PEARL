import os
import cv2
import shutil
import argparse

from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from ...constant import EVAL_DATASET_HF_PATH
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import process_video_frames, load_frames, qa_template


TASK_INST = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer."


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']

    query_inputs, cand_inputs, dataset_infos = [], [], []

    for video_id, question, answer, qid, _type, a0, a1, a2, a3, a4 in zip(
        batch_dict['video'], 
        batch_dict['question'], 
        batch_dict['answer'], 
        batch_dict['qid'], 
        batch_dict['type'], 
        batch_dict['a0'], 
        batch_dict['a1'], 
        batch_dict['a2'], 
        batch_dict['a3'], 
        batch_dict['a4']
    ):
        options = [a0, a1, a2, a3, a4]
        
        # Process paths
        video_path = f'{video_root}/{video_id}.mp4'
        frame_dir = f'{frame_root}/{video_id}'

        # Extract video frames (atomic operation to ensure local frames exist)
        if not os.path.exists(frame_dir) or not len(os.listdir(frame_dir)):
            os.makedirs(frame_dir, exist_ok=True)
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total_frames // max_frames_saved)
                saved_frames = 0
                while saved_frames < max_frames_saved:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_frames * step)
                    ret, frame = cap.read()
                    if not ret: break
                    cv2.imwrite(os.path.join(frame_dir, f"{saved_frames:04d}.jpeg"), frame)
                    saved_frames += 1
                cap.release()

        # Format question text using qa_template (automatically appends A. B. C. D. E.)
        formatted_question, formatted_cands, _, _ = qa_template(question, options, answer)

        # Build inputs
        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query: video frame paths + formatted question (with options)
        query_inputs.append({
            "video": qry_frame_paths,
            "text": formatted_question,
            "instruction": TASK_INST,
        })

        # Candidates: 5 options as separate text embeddings
        cand_inputs.append([{"text": opt} for opt in formatted_cands])

        dataset_infos.append({
            "question_id": qid,
            "video_id": video_id,
            "query": formatted_question,
            "cand_names": formatted_cands,
            "label_name": formatted_cands[answer],
            "answer": options[answer],
            "answer_idx": answer,
            "type": _type,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "nextqa"

@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_nextqa_dataset(model_args, data_args, *args, **kwargs):
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs), 
        batched=True,
        batch_size=256, 
        num_proc=4,
        drop_last_batch=False, 
        load_from_cache_file=False
    )
    
    dataset = dataset.select_columns(["query_input", "cand_input", "dataset_infos"])

    return dataset, None

def export_dataset_for_hf(
    output_dir: str,
    dataset_name: str = "nextqa",
    max_queries: int = None,
    max_corpus: int = None,
    video_root: str = None,
    frame_root: str = None,
    max_frames_saved: int = 32,
    num_frames: int = 8,
):
    """
    Export NextQA dataset in a format loadable by datasets.load_dataset
    
    Args:
        output_dir: Save directory
        dataset_name: Dataset name
        max_queries: Maximum number of queries
        max_corpus: Maximum number of corpus items
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
    
    dataset, _ = load_nextqa_dataset(None, None, **kwargs)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Prepare three subsets
    queries_data = []
    corpus_data = {}  # Use dict for deduplication
    relevant_docs_data = []
    
    corpus_id_counter = 0
    corpus_text_to_id = {}  # Text to ID mapping for deduplication
    
    print("Processing dataset...")
    for idx, sample in enumerate(tqdm(dataset)):
        query_input = sample['query_input']
        cand_input = sample['cand_input']
        dataset_info = sample['dataset_infos']
        
        # Process video files
        video_id = dataset_info['video_id']
        query_id = f"q_{idx}"
        
        # Copy video frames to output directory
        video_frames = query_input['video']
        saved_video_paths = []
        for frame_idx, frame_path in enumerate(video_frames):
            if os.path.exists(frame_path):
                target_frame_path = videos_dir / f"{video_id}_{frame_idx:04d}.jpeg"
                if not target_frame_path.exists():
                    shutil.copy2(frame_path, target_frame_path)
                saved_video_paths.append(str(target_frame_path.relative_to(output_path)))
        
        # Build queries
        queries_data.append({
            "query_id": query_id,
            "query_text": query_input['text'],
            "instruction": query_input['instruction'],
            "video_paths": saved_video_paths,
            "video_id": video_id,
            "question_id": dataset_info['question_id'],
            "type": dataset_info['type'],
        })
        
        # Build corpus (candidate answers)
        positive_corpus_ids = []
        for cand_idx, cand in enumerate(cand_input):
            cand_text = cand['text']
            
            # Deduplication: if text exists, reuse existing ID
            if cand_text in corpus_text_to_id:
                corpus_id = corpus_text_to_id[cand_text]
            else:
                corpus_id = f"c_{corpus_id_counter}"
                corpus_id_counter += 1
                corpus_text_to_id[cand_text] = corpus_id
                
                corpus_data[corpus_id] = {
                    "corpus_id": corpus_id,
                    "text": cand_text,
                }
            
            # Record if correct answer
            if cand_idx == dataset_info['answer_idx']:
                positive_corpus_ids.append(corpus_id)
        
        # Build relevant_docs
        relevant_docs_data.append({
            "query_id": query_id,
            "corpus_ids": positive_corpus_ids,
        })
        
        # Limit corpus size
        if max_corpus and len(corpus_data) >= max_corpus:
            break
    
    # Convert to Dataset
    print("Creating HuggingFace datasets...")
    queries_dataset = Dataset.from_list(queries_data)
    corpus_dataset = Dataset.from_list(list(corpus_data.values()))
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
    print(f"Output directory: {output_dir}")
    print(f"Queries: {len(queries_dataset)}")
    print(f"Corpus: {len(corpus_dataset)}")
    print(f"Relevant docs: {len(relevant_docs_dataset)}")
    print(f"Videos directory: {videos_dir}")
    print("="*50)
    
    # Save loading instructions
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# NextQA Dataset Export

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
  - query_text: formatted question with options
  - instruction: task instruction
  - video_paths: list of video frame paths
  - video_id: original video ID
  - question_id: original question ID
  - type: question type

- **corpus**: {len(corpus_dataset)} samples
  - corpus_id: unique corpus identifier
  - text: answer option text

- **relevant_docs**: {len(relevant_docs_dataset)} samples
  - query_id: corresponding query ID
  - corpus_ids: list of correct answer corpus IDs

## Videos

Video frames are stored in: `{videos_dir.relative_to(output_path)}/`
""")
    
    print(f"\nREADME saved to: {readme_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export NextQA dataset for HuggingFace datasets"
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
        default="nextqa",
        help="Name of the dataset to load (default: nextqa)"
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
        help="Maximum number of corpus items to export (default: all)"
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
    
    export_dataset_for_hf(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_queries=args.max_queries,
        max_corpus=args.max_corpus,
        video_root=args.video_root,
        frame_root=args.frame_root,
        max_frames_saved=args.max_frames_saved,
        num_frames=args.num_frames,
    )