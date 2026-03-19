import os

from datasets import load_dataset
from ...constant import EVAL_DATASET_HF_PATH
from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import save_frames, load_frames


TASK_INST_QRY_TEXT = "Find the clip that corresponds to the given text."
TASK_INST_QRY_IMG = "Select the video clip that aligns with the given text and image."
TASK_INST_QRY_VIDEO = "Find the clip that corresponds to the given sentence and video segment."
# TASK_INST_TGT = "Understand the content of the provided video clip."


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    num_video_frames = kwargs["num_video_frames"]
    video_root = kwargs["video_root"]
    clip_root = kwargs["clip_root"]
    frame_root = kwargs["frame_root"]

    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for query, positive_frames, negative_frames, input_frames in zip(
        batch_dict['query'],
        batch_dict["positive_frames"],
        batch_dict["negative_frames"],
        batch_dict["input_frames"]
    ):
        if input_frames.endswith(".mp4"):
            query_video_name = input_frames.split(".mp4")[0].replace("/", "_")
            query_frame_dir = os.path.join(frame_root, "video_frames", query_video_name)
            if not os.path.exists(query_frame_dir):
                query_video_path = os.path.join(video_root, input_frames)
                save_frames(
                    video_path=query_video_path,
                    frame_dir=query_frame_dir,
                    max_frames_saved=num_video_frames
                )
            qry_frame_paths = load_frames(query_frame_dir)
            query_inputs.append({
                "text": query,
                "video": qry_frame_paths,
                "instruction": TASK_INST_QRY_VIDEO,
            })
        elif input_frames.endswith(".jpg"):
            input_image_path = os.path.join(frame_root, "", f"query_{input_frames}")
            query_inputs.append({
                "text": query,
                "image": input_image_path,
                "instruction": TASK_INST_QRY_IMG,
            })
        else:
            query_inputs.append({
                "text": query,
                "instruction": TASK_INST_QRY_TEXT,
            })

        pos_clip_paths = [entry["output_path"] for entry in positive_frames]
        neg_clip_paths = [entry["output_path"] for entry in negative_frames]

        pos_clip_name = []
        cand_clip_names = []
        cand_list = []
        
        for path in pos_clip_paths:
            cand_clip_name = path.replace("/", "_").split(".mp4")[0]
            cand_clip_frame_dir = os.path.join(frame_root, "video_frames", cand_clip_name)
            if not os.path.exists(cand_clip_frame_dir):
                cand_clip_abs_path = os.path.join(clip_root, path)
                save_frames(
                    video_path=cand_clip_abs_path,
                    frame_dir=cand_clip_frame_dir,
                    max_frames_saved=num_video_frames
                )
            pos_clip_frames = load_frames(cand_clip_frame_dir)
            cand_list.append({
                "video": pos_clip_frames,
                # "instruction": TASK_INST_TGT,
            })
            cand_clip_names.append(cand_clip_frame_dir)
            pos_clip_name.append(cand_clip_frame_dir)
        
        for path in neg_clip_paths:
            cand_clip_name = path.replace("/", "_").split(".mp4")[0]
            cand_clip_frame_dir = os.path.join(frame_root, "video_frames", cand_clip_name)
            if not os.path.exists(cand_clip_frame_dir):
                cand_clip_abs_path = os.path.join(clip_root, path)
                save_frames(
                    video_path=cand_clip_abs_path,
                    frame_dir=cand_clip_frame_dir,
                    max_frames_saved=num_video_frames
                )
            neg_clip_frames = load_frames(cand_clip_frame_dir)
            cand_list.append({
                "video": neg_clip_frames,
                # "instruction": TASK_INST_TGT,
            })
            cand_clip_names.append(cand_clip_frame_dir)

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


DATASET_PARSER_NAME = "momentseeker"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_momentseeker_dataset(model_args, data_args, *args, **kwargs):
    if kwargs.get("data_path", None) is not None:
        dataset = load_dataset("json", data_files=kwargs["data_path"])
        dataset = dataset["train"]
    else:
        dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['global_dataset_name'] = kwargs['dataset_name'] if kwargs['dataset_name'] else DATASET_PARSER_NAME

    dataset = dataset.map(
        lambda x: data_prepare(x, **kwargs),
        batched=True,
        batch_size=2048,
        num_proc=1,
        drop_last_batch=False,
        load_from_cache_file=False
    )
    dataset = dataset.select_columns(["query_input", "cand_input", "dataset_infos"])
    return dataset, None