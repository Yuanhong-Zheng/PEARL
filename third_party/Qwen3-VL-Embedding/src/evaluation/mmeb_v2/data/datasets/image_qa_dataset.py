import os
import sys

from datasets import load_dataset
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_root = kwargs['image_root']

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for qry_inst, qry_text, qry_img_path, tgt_texts in zip(
        batch_dict['qry_inst'], 
        batch_dict['qry_text'], 
        batch_dict['qry_img_path'], 
        batch_dict['tgt_text']
    ):
        clean_inst = qry_inst.replace("<|image_1|>", "").strip()
        full_img_path = os.path.join(image_root, qry_img_path)

        query_inputs.append({
            "image": full_img_path,
            "text": qry_text,
            "instruction": clean_inst,
        })

        cand_inputs.append([{"text": t} for t in tgt_texts])
        
        dataset_infos.append({
            "cand_names": tgt_texts,
            "label_name": tgt_texts[0],
        })

    return {
        "query_input": query_inputs, 
        "cand_input": cand_inputs, 
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "image_qa"
DATASET_HF_PATH = "ziyjiang/MMEB_Test_Instruct"

@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_image_qa_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]
    dataset = load_dataset(DATASET_HF_PATH, dataset_name, split="test")
    
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if num_sample_per_subset is not None and isinstance(num_sample_per_subset, str) and num_sample_per_subset.isdigit():
        num_sample_per_subset = int(num_sample_per_subset)
    
    if num_sample_per_subset < dataset.num_rows:
        dataset = dataset.select(range(num_sample_per_subset))
        print(f"Subsample to {len(dataset)} samples")

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