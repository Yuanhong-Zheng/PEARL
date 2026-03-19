import os

from datasets import load_dataset
from ...constant import EVAL_DATASET_HF_PATH
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import save_frames, process_video_frames, VID_EXTENSIONS


TASK_INST_QRY = "Find the clip that corresponds to the described scene in the given video."
# TASK_INST_TGT = "Understand the content of the provided video."


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    max_video_frames_saved = kwargs["max_video_frames_saved"]
    max_clip_frames_saved = kwargs["max_clip_frames_saved"]
    num_video_frames = kwargs["num_video_frames"]
    num_clip_frames = kwargs["num_clip_frames"]
    video_root = kwargs["video_root"]
    clip_root = kwargs["clip_root"]
    frame_root = kwargs["frame_root"]

    query_inputs, cand_inputs, dataset_infos = [], [], []

    for query, query_video_path in zip(batch_dict['query'], batch_dict['video_path']):
        video_name = os.path.splitext(os.path.basename(query_video_path))[0]
        frames_dir = os.path.join(frame_root, video_name)

        query_video_path = os.path.join(video_root, os.path.basename(query_video_path)) if video_root else None
        query_frame_dir = os.path.join(frames_dir, "query")
        if not os.path.exists(query_frame_dir):
            save_frames(
                video_path=query_video_path,
                frame_dir=query_frame_dir,
                max_frames_saved=max_video_frames_saved
            )
        qry_frame_paths = process_video_frames(query_frame_dir, num_frames=num_video_frames)

        query_inputs.append({
            "text": query,
            "video": qry_frame_paths,
            "instruction": TASK_INST_QRY,
        })

        # Save frames from raw video if not already extracted
        if not os.path.exists(frames_dir):
            clip_video_dir = os.path.join(clip_root, video_name) if clip_root else None
            clip_video_paths = [f for f in os.listdir(clip_video_dir) if os.path.splitext(f)[1].lower() in VID_EXTENSIONS]
            for clip_video_path in clip_video_paths:
                clip_name = os.path.splitext(clip_video_path)[0]
                clip_frame_dir_or_file = os.path.join(frames_dir, clip_name)
                clip_video_path_abs = os.path.join(clip_video_dir, clip_video_path)
                save_frames(
                    video_path=clip_video_path_abs,
                    frame_dir=clip_frame_dir_or_file,
                    max_frames_saved=max_clip_frames_saved
                )

        cand_clip_names = []
        cand_list = []
        pos_clip_name = None
        
        for clip_frame_dir_or_file in os.listdir(frames_dir):
            clip_frame_dir_abs = os.path.join(frames_dir, clip_frame_dir_or_file)
            if clip_frame_dir_or_file == 'query' or os.path.isfile(clip_frame_dir_abs):
                continue
            if clip_frame_dir_or_file.startswith("positive"):
                pos_clip_name = clip_frame_dir_abs
            
            cand_frame_paths = process_video_frames(clip_frame_dir_abs, num_frames=num_clip_frames)
            cand_list.append({
                "video": cand_frame_paths,
                # "instruction": TASK_INST_TGT,
            })
            cand_clip_names.append(clip_frame_dir_abs)

        cand_inputs.append(cand_list)
        dataset_infos.append({
            "cand_names": cand_clip_names,
            "label_name": pos_clip_name,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "moment_retrieval"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_moment_retrieval_dataset(model_args, data_args, **kwargs):
    if kwargs.get("data_path", None) is not None:
        dataset = load_dataset("json", data_files=kwargs["data_path"])
        dataset = dataset["train"]
    else:
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=2048,
        num_proc=8,
        drop_last_batch=False,
        load_from_cache_file=False
    )
    dataset = dataset.select_columns(["query_input", "cand_input", "dataset_infos"])
    corpus = None

    return dataset, corpus