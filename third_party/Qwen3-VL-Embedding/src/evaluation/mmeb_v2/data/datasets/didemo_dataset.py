import os

from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...constant import EVAL_DATASET_HF_PATH
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import process_video_frames, save_frames


TASK_INST_QRY = "Find a video that includes the following described scenes."
# TASK_INST_TGT = "Understand the content of the provided video."


@add_metainfo_hook
def data_prepare(batch_dict, **kwargs):
    num_frames = kwargs['num_frames']
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for video_path, caption in zip(batch_dict['video'], batch_dict['caption']):
        query_inputs.append({
            "text": caption,
            "instruction": TASK_INST_QRY,
        })

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_path = os.path.join(video_root, os.path.basename(video_path))
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


DATASET_PARSER_NAME = "didemo"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_didemo_dataset(model_args, data_args, **kwargs):
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