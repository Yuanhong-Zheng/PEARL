import os
import sys

from datasets import load_dataset
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_root = kwargs['image_root']
    dataset_name = kwargs.get('dataset_name', '')

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for qry_inst, qry_text, tgt_inst, tgt_captions, tgt_img_paths in zip(
        batch_dict['qry_inst'], 
        batch_dict['qry_text'], 
        batch_dict['tgt_inst'], 
        batch_dict['tgt_text'], 
        batch_dict['tgt_img_path']
    ):
        clean_qry_inst = qry_inst.replace("<|image_1|>", "").strip()
        
        query_inputs.append({
            "text": qry_text,
            "instruction": clean_qry_inst,
        })

        clean_tgt_inst = tgt_inst.replace("<|image_1|>", "").strip()
        current_cand_list = []

        for i, tgt_img_path in enumerate(tgt_img_paths):
            full_img_path = os.path.join(image_root, tgt_img_path)
            
            cand_item = {
                "image": full_img_path,
                # "instruction": clean_tgt_inst
            }
            
            # Add caption if available (for WebQA, EDIS, etc.)
            if i < len(tgt_captions) and tgt_captions[i].strip():
                cand_item["text"] = tgt_captions[i].strip()
            
            current_cand_list.append(cand_item)

        cand_inputs.append(current_cand_list)
        
        dataset_infos.append({
            "cand_names": tgt_img_paths,
            "label_name": tgt_img_paths[0],
        })

    return {
        "query_input": query_inputs, 
        "cand_input": cand_inputs, 
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "image_t2i"
DATASET_HF_PATH = "ziyjiang/MMEB_Test_Instruct"

@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_image_t2i_dataset(model_args, data_args, *args, **kwargs):
    dataset_name = kwargs["dataset_name"]
    dataset = load_dataset(DATASET_HF_PATH, dataset_name, split="test")
    
    num_sample_per_subset = kwargs.get("num_sample_per_subset", sys.maxsize)
    if isinstance(num_sample_per_subset, str) and num_sample_per_subset.isdigit():
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