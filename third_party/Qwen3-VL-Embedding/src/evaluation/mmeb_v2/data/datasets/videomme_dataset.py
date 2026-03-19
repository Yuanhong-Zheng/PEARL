import os
import cv2

from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...constant import EVAL_DATASET_HF_PATH
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import process_video_frames, load_frames


TASK_INST = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer."
OPTIONS_MAP = ['A', 'B', 'C', 'D']


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']

    query_inputs, cand_inputs, dataset_infos = [], [], []

    for query_text, video_id, options, answer, question_id, domain, sub_category in zip(
        batch_dict['question'], 
        batch_dict['videoID'], 
        batch_dict['options'], 
        batch_dict['answer'], 
        batch_dict['question_id'], 
        batch_dict['domain'], 
        batch_dict['sub_category']
    ):
        # Process paths
        video_path = f'{video_root}/{video_id}.mp4'
        frame_dir = f'{frame_root}/{video_id}'

        # Extract video frames (ensure local frames exist)
        if not os.path.exists(frame_dir) or not len(os.listdir(frame_dir)):
            os.makedirs(frame_dir, exist_ok=True)
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total_frames // max_frames_saved)
                saved_frames = 0
                while saved_frames < max_frames_saved:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_frames * step)
                    ret, frame = cap.read()
                    if not ret: break
                    cv2.imwrite(os.path.join(frame_dir, f"{saved_frames:04d}.jpeg"), frame)
                    saved_frames += 1
                cap.release()

        # Text processing
        # Build query text: Question + Options
        full_query_text = query_text + '\n' + '\n'.join(options)
        
        # Build clean candidate text (remove option prefixes like "A. ")
        cleaned_cands = [o[o.find('. '):].strip('. ') if '. ' in o else o for o in options]

        # Get frame paths
        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Build input structure
        # Query: video frames + question text with options
        query_inputs.append({
            "video": qry_frame_paths,
            "text": full_query_text,
            "instruction": TASK_INST,
        })

        # Candidates: clean option text list
        cand_inputs.append([{"text": c} for c in cleaned_cands])

        # Save metadata
        answer_idx = OPTIONS_MAP.index(answer)
        dataset_infos.append({
            "question_id": question_id,
            "video_id": video_id,
            "query": full_query_text,
            "cand_names": options,
            "answer": answer,
            "label_name": options[answer_idx],
            "answer_idx": answer_idx,
            "domain": domain,
            "sub_category": sub_category,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos
    }


DATASET_PARSER_NAME = "videomme"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_videomme_dataset(model_args, data_args, *args, **kwargs):
    dataset = load_hf_dataset(EVAL_DATASET_HF_PATH[kwargs['dataset_name']])
    dataset = sample_dataset(dataset, **kwargs)

    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME

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