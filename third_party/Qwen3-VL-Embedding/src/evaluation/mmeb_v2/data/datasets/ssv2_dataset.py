import os

from .video_classification_utils import DATASET_INSTRUCTION
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

    # Get task instruction for this dataset
    instruction = DATASET_INSTRUCTION.get(dataset_name, "Understand the action in the video.")
    # Normalize instruction punctuation (remove colon, replace with period)
    if instruction.endswith(":"):
        instruction = instruction[:-1] + "."

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for video_id, pos_text, cand_text in zip(
        batch_dict['video_id'], 
        batch_dict['pos_text'], 
        batch_dict['neg_text']
    ):
        # Process video path and extract frames
        video_path = os.path.join(video_root, str(video_id) + '.mp4')
        frame_dir = os.path.join(frame_root, str(video_id))
        if not os.path.exists(frame_dir):
            save_frames(video_path=video_path, frame_dir=frame_dir, max_frames_saved=max_frames_saved)
        video_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Query input: video as primary input with instruction describing the task
        query_inputs.append({
            "video": video_frame_paths,
            "instruction": instruction,
        })

        # Candidate input: plain text list (all candidate action descriptions including positive and negative samples)
        # In SSv2-MC (Multiple Choice) mode, cand_text typically contains multiple alternative actions
        cand_inputs.append([{"text": t} for t in cand_text])
        
        dataset_infos.append({
            "cand_names": cand_text,
            "label_name": pos_text,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "ssv2"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_ssv2_dataset(model_args, data_args, **kwargs):
    """
    SSv2-MC setup for zero-shot evaluation.
    """
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
    corpus = None

    return dataset, corpus