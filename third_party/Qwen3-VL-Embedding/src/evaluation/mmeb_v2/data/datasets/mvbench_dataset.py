import os
import shutil
import cv2

from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...constant import EVAL_DATASET_HF_PATH
from ...utils.dataset_utils import load_hf_dataset_multiple_subset, sample_dataset
from ...utils.vision_utils.vision_utils import process_video_frames, qa_template


# Subset metadata for path mapping
subset_meta = {
    "episodic_reasoning": {"video_path": "tvqa/frames_fps3_hq/", "data_type": "frame"},
    "action_sequence": {"video_path": "star/Charades_v1_480/", "data_type": "video"},
    "action_prediction": {"video_path": "star/Charades_v1_480/", "data_type": "video"},
    "action_antonym": {"video_path": "ssv2_video/", "data_type": "video"},
    "fine_grained_action": {"video_path": "Moments_in_Time_Raw/videos/", "data_type": "video"},
    "unexpected_action": {"video_path": "FunQA_test/test/", "data_type": "video"},
    "object_existence": {"video_path": "clevrer/video_validation/", "data_type": "video"},
    "object_interaction": {"video_path": "star/Charades_v1_480/", "data_type": "video"},
    "object_shuffle": {"video_path": "perception/videos/", "data_type": "video"},
    "moving_direction": {"video_path": "clevrer/video_validation/", "data_type": "video"},
    "action_localization": {"video_path": "sta/sta_video/", "data_type": "video"},
    "scene_transition": {"video_path": "scene_qa/video/", "data_type": "video"},
    "action_count": {"video_path": "perception/videos/", "data_type": "video"},
    "moving_count": {"video_path": "clevrer/video_validation/", "data_type": "video"},
    "moving_attribute": {"video_path": "clevrer/video_validation/", "data_type": "video"},
    "state_change": {"video_path": "perception/videos/", "data_type": "video"},
    "fine_grained_pose": {"video_path": "nturgbd/", "data_type": "video"},
    "character_order": {"video_path": "perception/videos/", "data_type": "video"},
    "egocentric_navigation": {"video_path": "vlnqa/", "data_type": "video"},
    "counterfactual_inference": {"video_path": "clevrer/video_validation/", "data_type": "video"}
}

TASK_INST = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer."


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']

    query_inputs, cand_inputs, dataset_infos = [], [], []

    for subset, question, video_filename, cands, answer in zip(
        batch_dict['subset'], 
        batch_dict['question'], 
        batch_dict['video'], 
        batch_dict['candidates'], 
        batch_dict['answer']
    ):
        subset_meta_info = subset_meta[subset]
        data_type = subset_meta_info["data_type"]
        
        # Process paths
        video_path = f'{video_root}/{subset_meta_info["video_path"]}/{video_filename}'
        frame_dir = f'{frame_root}/{subset}/{video_filename}'

        # Save video as frames (keeping original logic to ensure local frame files exist)
        if data_type == "video" and (not os.path.exists(frame_dir) or not len(os.listdir(frame_dir))):
            os.makedirs(frame_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames_saved)
            saved_count = 0
            while saved_count < max_frames_saved:
                cap.set(cv2.CAP_PROP_POS_FRAMES, saved_count * step)
                ret, frame = cap.read()
                if not ret: break
                cv2.imwrite(os.path.join(frame_dir, f"{saved_count:04d}.jpeg"), frame)
                saved_count += 1
            cap.release()
        elif data_type == "frame" and (not os.path.exists(frame_dir) or not len(os.listdir(frame_dir))):
            shutil.copytree(video_path, frame_dir, dirs_exist_ok=True)

        # Format question and options using qa_template
        formatted_query, formatted_cands, formatted_answer, answer_idx = qa_template(question, cands, answer)
        
        # Build inputs
        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        
        # Query: video + formatted question text
        query_inputs.append({
            "video": qry_frame_paths,
            "text": formatted_query,
            "instruction": TASK_INST,
        })

        # Candidates: plain text option list
        cand_inputs.append([{"text": c} for c in formatted_cands])

        dataset_infos.append({
            "subset": subset,
            "video_id": video_filename,
            "query": formatted_query,
            "cand_names": formatted_cands,
            "answer": formatted_answer,
            "label_name": formatted_answer,
            "answer_idx": answer_idx,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "mvbench"

@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_mvbench_dataset(model_args, data_args, *args, **kwargs):
    dataset = load_hf_dataset_multiple_subset(
        EVAL_DATASET_HF_PATH[kwargs['dataset_name']], 
        subset_meta.keys()
    )
    dataset = sample_dataset(dataset, **kwargs)
    
    kwargs['global_dataset_name'] = kwargs['dataset_name'] if kwargs['dataset_name'] else DATASET_PARSER_NAME

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