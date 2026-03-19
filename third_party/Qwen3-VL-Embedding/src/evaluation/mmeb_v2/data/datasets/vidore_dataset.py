import os
import argparse
import shutil
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...constant import EVAL_DATASET_HF_PATH
from ...utils.dataset_utils import load_hf_dataset, sample_dataset, load_qrels_mapping
from ...utils.basic_utils import print_master


TASK_INST_QRY = "Find a document image that matches the given query."
# TASK_INST_TGT = "Understand the content of the provided document image."


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    qrels_mapping = kwargs['qrels_mapping']
    image_root = kwargs['image_root']

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for query_id, query in zip(batch_dict['query-id'], batch_dict['query']):
        # Query: plain text
        query_inputs.append({
            "text": query,
            "instruction": TASK_INST_QRY,
        })
        
        cand_list, cand_names, label_names, rel_scores = [], [], [], []

        # Find relevant candidate images from qrels mapping
        for corpus_id, rel_score in qrels_mapping[query_id].items():
            image_path = f'{image_root}/{corpus_id}.png'
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'Image path {image_path} not found.')
            
            # Candidate: document image
            cand_list.append({
                "image": image_path,
                # "instruction": TASK_INST_TGT,
            })
            cand_names.append(corpus_id)
            label_names.append(corpus_id)
            rel_scores.append(rel_score)
            
        cand_inputs.append(cand_list)
        dataset_infos.append({
            "cand_names": cand_names,
            "label_name": label_names,
            "rel_scores": rel_scores,
        })

    return {
        "query_input": query_inputs, 
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos
    }


def corpus_prepare(batch_dict, *args, **kwargs):
    image_root = kwargs['image_root']

    cand_inputs, dataset_infos = [], []
    for corpus_id, image in zip(batch_dict['corpus-id'], batch_dict['image']):
        image_path = f'{image_root}/{corpus_id}.png'
        if not os.path.exists(image_path):
            os.makedirs(image_root, exist_ok=True)
            image.save(image_path)
            
        cand_inputs.append([{
            "image": image_path,
            # "instruction": TASK_INST_TGT,
        }])
        
        dataset_infos.append({
            "cand_names": [corpus_id],
        })

    return {
        "cand_input": cand_inputs, 
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "vidore"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_vidore_dataset(model_args, data_args, **kwargs):
    hf_dataset_name = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][0]
    hf_dataset_split = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][2]
    lang = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][1]
    
    # Load BEIR format subsets
    dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
    if lang is not None:
        dataset = dataset.filter(lambda example: example["language"] == lang)
    
    qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
    corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
    qrels_mapping = load_qrels_mapping(qrels)
    
    dataset = sample_dataset(dataset, **kwargs)
    print_master(f"Loaded {kwargs['dataset_name']}")

    kwargs['qrels_mapping'] = qrels_mapping

    # Process corpus
    corpus = corpus.map(
        lambda x: corpus_prepare(x, **kwargs), 
        batched=True,
        batch_size=2048, 
        num_proc=8,
        drop_last_batch=False, 
        load_from_cache_file=False
    )
    corpus = corpus.select_columns(['cand_input', 'dataset_infos'])
    
    # Process queries
    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs), 
        batched=True,
        batch_size=2048, 
        num_proc=8,
        drop_last_batch=False, 
        load_from_cache_file=False
    )
    dataset = dataset.select_columns(["query_input", "cand_input", "dataset_infos"])

    return dataset, corpus


def export_vidore_dataset_for_hf(
    output_dir: str,
    dataset_name: str,
    max_queries: int = None,
    max_corpus: int = None,
    image_root: str = None,
):
    """
    Export ViDoRe text-to-document image retrieval dataset for HuggingFace datasets.load_dataset
    
    Args:
        output_dir: Output directory path
        dataset_name: Dataset name
        max_queries: Maximum number of queries to export
        max_corpus: Maximum number of corpus items (document images) to export
        image_root: Root directory for image files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"Loading dataset: {dataset_name}")
    hf_dataset_name = EVAL_DATASET_HF_PATH[dataset_name][0]
    hf_dataset_split = EVAL_DATASET_HF_PATH[dataset_name][2]
    lang = EVAL_DATASET_HF_PATH[dataset_name][1]
    
    kwargs = {
        'dataset_name': dataset_name,
        'image_root': image_root or './images',
    }
    
    if max_queries:
        kwargs['sample_size'] = max_queries
    
    dataset, corpus = load_vidore_dataset(None, None, **kwargs)
    
    print(f"Dataset loaded with {len(dataset)} queries and {len(corpus)} documents")
    
    queries_data = []
    corpus_data = {}  # Use dict for deduplication
    relevant_docs_data = []
    
    print("Processing corpus (document images)...")
    # Process corpus first to build corpus_id mapping
    for idx, corpus_item in enumerate(tqdm(corpus)):
        if max_corpus and idx >= max_corpus:
            break
            
        cand_input = corpus_item['cand_input'][0]
        dataset_info = corpus_item['dataset_infos']
        corpus_id = dataset_info['cand_names'][0]
        
        # Copy image to output directory
        image_path = cand_input['image']
        if os.path.exists(image_path):
            target_image_path = images_dir / f"{corpus_id}.png"
            if not target_image_path.exists():
                shutil.copy2(image_path, target_image_path)
            
            corpus_data[corpus_id] = {
                "corpus_id": corpus_id,
                "image_path": str(target_image_path.relative_to(output_path)),
            }
    
    print(f"Processed {len(corpus_data)} corpus items")
    
    print("Processing queries...")
    processed_queries = 0
    for idx, sample in enumerate(tqdm(dataset)):
        if max_queries and processed_queries >= max_queries:
            break
            
        query_input = sample['query_input']
        dataset_info = sample['dataset_infos']
        
        query_id = f"q_{idx}"
        
        queries_data.append({
            "query_id": query_id,
            "text": query_input['text'],
            "instruction": query_input['instruction'],
        })
        
        relevant_corpus_ids = []
        relevance_scores = []
        
        for cand_name, rel_score in zip(dataset_info['cand_names'], dataset_info['rel_scores']):
            # Only include documents present in corpus
            if cand_name in corpus_data:
                relevant_corpus_ids.append(cand_name)
                relevance_scores.append(rel_score)
        
        # Only add query if it has at least one relevant document in corpus
        if relevant_corpus_ids:
            relevant_docs_data.append({
                "query_id": query_id,
                "corpus_ids": relevant_corpus_ids,
                "scores": relevance_scores,
            })
            processed_queries += 1
        else:
            queries_data.pop()
    
    print(f"Processed {len(queries_data)} queries with relevant documents")
    
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
    
    # Print statistics
    print("\n" + "="*50)
    print("Dataset Export Summary")
    print("="*50)
    print(f"Dataset name: {dataset_name}")
    print(f"Language filter: {lang if lang else 'None'}")
    print(f"Output directory: {output_dir}")
    print(f"Queries: {len(queries_dataset)}")
    print(f"Corpus (document images): {len(corpus_dataset)}")
    print(f"Relevant docs: {len(relevant_docs_dataset)}")
    print(f"Images directory: {images_dir}")
    print(f"Task: Text-to-Document Image Retrieval")
    
    if relevant_docs_data:
        avg_relevant = sum(len(rd['corpus_ids']) for rd in relevant_docs_data) / len(relevant_docs_data)
        print(f"Average relevant docs per query: {avg_relevant:.2f}")
    
    print("="*50)
    
    # Save README
    readme_path = output_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"""# ViDoRe Text-to-Document Image Retrieval Dataset Export

## Dataset Information

- **Dataset**: {dataset_name}
- **Source**: {hf_dataset_name}
- **Split**: {hf_dataset_split}
- **Language**: {lang if lang else 'All languages'}

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

- **queries**: {len(queries_dataset)} samples (text queries)
  - query_id: unique query identifier
  - text: text query for document retrieval
  - instruction: retrieval task instruction ("{TASK_INST_QRY}")

- **corpus**: {len(corpus_dataset)} samples (document images)
  - corpus_id: unique corpus identifier
  - image_path: path to document image file (relative to dataset root)

- **relevant_docs**: {len(relevant_docs_dataset)} samples
  - query_id: corresponding query ID
  - corpus_ids: list of relevant document corpus IDs
  - scores: relevance scores for each document (higher is more relevant)

## Task Description

**Task Type**: Text-to-Document Image Retrieval

**Instruction**: {TASK_INST_QRY}

This is a text-to-document image retrieval task where text queries are matched to relevant document images from the corpus. Documents may have varying degrees of relevance indicated by the relevance scores.

## Images

Document images are stored in: `{images_dir.relative_to(output_path)}/`

All images are in PNG format.

## Statistics

- Total document images: {len(corpus_dataset)}
- Total text queries: {len(queries_dataset)}
- Average relevant documents per query: {sum(len(rd['corpus_ids']) for rd in relevant_docs_data) / len(relevant_docs_data):.2f}

## Relevance Scores

The `scores` field in `relevant_docs` contains relevance judgments:
- Higher scores indicate higher relevance
- Scores typically range from 0 to 3 or similar scale
- Use these scores for evaluation metrics like NDCG

## Example Usage

```python
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk("{output_dir}")
queries = dataset["queries"]
corpus = dataset["corpus"]
relevant_docs = dataset["relevant_docs"]

# Get a query and its relevant documents
query = queries[0]
print(f"Query: {{query['text']}}")

# Find relevant documents
relevant = relevant_docs[0]
print(f"Relevant document IDs: {{relevant['corpus_ids']}}")
print(f"Relevance scores: {{relevant['scores']}}")

# Load relevant document images
for corpus_id, score in zip(relevant['corpus_ids'], relevant['scores']):
    doc = corpus.filter(lambda x: x['corpus_id'] == corpus_id)[0]
    print(f"Document {{corpus_id}} (score={{score}}): {{doc['image_path']}}")
```

## Citation

Please cite the original ViDoRe dataset:

```bibtex
@article{{faysse2024colpali,
  title={{ColPali: Efficient Document Retrieval with Vision Language Models}},
  author={{Faysse, Manuel and Laurençon, Hugues and others}},
  journal={{arXiv preprint}},
  year={{2024}}
}}
```
""")
    
    print(f"\nREADME saved to: {readme_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export ViDoRe Text-to-Document Image Retrieval dataset for HuggingFace datasets"
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
        help="Maximum number of corpus items (document images) to export (default: all)"
    )
    
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Root directory containing image files (default: ./images)"
    )
    
    args = parser.parse_args()
    
    export_vidore_dataset_for_hf(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        max_queries=args.max_queries,
        max_corpus=args.max_corpus,
        image_root=args.image_root,
    )