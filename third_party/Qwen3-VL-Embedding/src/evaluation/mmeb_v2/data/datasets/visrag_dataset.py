import os
import hashlib

from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...constant import EVAL_DATASET_HF_PATH
from ...utils.dataset_utils import load_hf_dataset, sample_dataset, load_qrels_mapping


TASK_INST_QRY = "Find a document image that matches the given query."
# TASK_INST_TGT = "Understand the content of the provided document image."


def get_short_imagename(image_name):
    """Truncate filename and append MD5 hash to ensure path compatibility"""
    base, ext = os.path.splitext(image_name)
    # Keep first 50 chars and append first 8 chars of MD5 hash
    short_base = base[:50] + "_" + hashlib.md5(image_name.encode('utf-8')).hexdigest()[:8]
    return short_base + ext


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

        # Iterate through qrels mapping
        for image_name, rel_score in qrels_mapping[query_id].items():
            new_imagename = get_short_imagename(image_name)
            image_path = f'{image_root}/{new_imagename}'
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f'Image path {image_path} not found.')
            
            # Candidate: document image
            cand_list.append({
                "image": image_path,
                # "instruction": TASK_INST_TGT,
            })
            cand_names.append(image_name)
            label_names.append(image_name)
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
    for image_name, image in zip(batch_dict['corpus-id'], batch_dict['image']):
        new_imagename = get_short_imagename(image_name)
        image_path = f'{image_root}/{new_imagename}'
        
        if not os.path.exists(image_path):
            os.makedirs(image_root, exist_ok=True)
            image.save(image_path)
            
        cand_inputs.append([{
            "image": image_path,
            # "instruction": TASK_INST_TGT,
        }])
        dataset_infos.append({
            "cand_names": [image_name],
        })

    return {
        "cand_input": cand_inputs, 
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "visrag"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_visrag_dataset(model_args, data_args, **kwargs):
    hf_dataset_name = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][0]
    hf_dataset_split = EVAL_DATASET_HF_PATH[kwargs['dataset_name']][2]
    
    # Load BEIR format data
    qrels = load_hf_dataset((hf_dataset_name, "qrels", hf_dataset_split))
    corpus = load_hf_dataset((hf_dataset_name, "corpus", hf_dataset_split))
    dataset = load_hf_dataset((hf_dataset_name, "queries", hf_dataset_split))
    
    qrels_mapping = load_qrels_mapping(qrels)
    dataset = sample_dataset(dataset, **kwargs)
    
    kwargs['qrels_mapping'] = qrels_mapping

    # Process corpus
    corpus = corpus.map(
        lambda x: corpus_prepare(x, **kwargs), 
        batched=True,
        batch_size=1024, 
        num_proc=4,
        drop_last_batch=False, 
        load_from_cache_file=False
    )
    corpus = corpus.select_columns(['cand_input', 'dataset_infos'])
    
    # Process queries
    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs), 
        batched=True,
        batch_size=1024, 
        num_proc=4,
        drop_last_batch=False, 
        load_from_cache_file=False
    )
    dataset = dataset.select_columns(["query_input", "cand_input", "dataset_infos"])

    return dataset, corpus