import os
import sys

from datasets import load_dataset
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    image_root = kwargs['image_root']

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for qry_inst, qry_text, qry_img_path, tgt_inst, tgt_captions, tgt_img_paths in zip(
        batch_dict['qry_inst'],
        batch_dict['qry_text'],
        batch_dict['qry_img_path'],
        batch_dict['tgt_inst'],
        batch_dict['tgt_text'],
        batch_dict['tgt_img_path']
    ):
        qry_instruction = qry_inst.replace("<|image_1|>", "").strip()
        qry_img_path = os.path.join(image_root, qry_img_path)
        
        query_inputs.append({
            "text": qry_text,
            "image": qry_img_path,
            "instruction": qry_instruction,
        })

        tgt_instruction = tgt_inst.replace("<|image_1|>", "").strip()
        cand_img_paths = [os.path.join(image_root, p) for p in tgt_img_paths]
        
        # RefCOCO-Matching has valid text input
        if tgt_captions[0].strip():
            cand_inputs.append([{
                "text": tgt_cap,
                "image": cand_img_path,
                # "instruction": tgt_instruction,
            } for tgt_cap, cand_img_path in zip(tgt_captions, cand_img_paths)])
        else:  # Pure image matching
            cand_inputs.append([{
                "image": cand_img_path,
                # "instruction": tgt_instruction,
            } for cand_img_path in cand_img_paths])
        
        # Cand_names used for deduplication (same image may have multiple target objects in RefCOCO-Matching)
        cand_names = [path + ':' + cap.strip('"') for path, cap in zip(tgt_img_paths, tgt_captions)]
        dataset_infos.append({
            "cand_names": cand_names,
            "label_name": cand_names[0],
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "image_i2i_vg"
DATASET_HF_PATH = "ziyjiang/MMEB_Test_Instruct"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_image_i2i_vg_dataset(model_args, data_args, *args, **kwargs):
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

    corpus = None

    return dataset, corpus