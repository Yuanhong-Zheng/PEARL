import os
import cv2
import datasets
from datasets import load_dataset

from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...utils.dataset_utils import sample_dataset
from ...utils.vision_utils.vision_utils import process_video_frames, load_frames, qa_template

# Standardized task instruction
TASK_INST = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer."
SUBSET_NAMES = ['Perception', 'Comprehension', 'Adaptation']

@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']

    query_inputs, cand_inputs, dataset_infos = [], [], []

    for video_id, question, answer, question_type, options, subset in zip(
        batch_dict['id'], 
        batch_dict['question'], 
        batch_dict['answer'], 
        batch_dict['question_type'], 
        batch_dict['options'], 
        batch_dict['subset']
    ):
        # Filter out non-multiple-choice questions
        if question_type != 'multiple-choice':
            continue

        # Construct video and frame paths
        video_path = f'{video_root}/{subset}/{video_id}.mp4'
        frame_dir = f'{frame_root}/{subset}/{video_id}'

        # Extract frames from video if not already cached
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

        # Format question and answer candidates using qa_template
        formatted_question, formatted_cands, _, _ = qa_template(question, options, answer)

        # Get frame paths for the video
        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)

        # Build query input: video frames + formatted question text
        query_inputs.append({
            "video": qry_frame_paths,
            "text": formatted_question,
            "instruction": TASK_INST,
        })

        # Build candidate inputs: clean option text list
        cand_inputs.append([{"text": c} for c in formatted_cands])

        dataset_infos.append({
            "video_id": video_id,
            "query": formatted_question,
            "cand_names": formatted_cands,
            "answer": options[answer] if isinstance(answer, int) else answer,
            "label_name": answer,
            "answer_idx": answer,
            "subset": subset,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos
    }

DATASET_PARSER_NAME = "videommmu"
DATASET_HF_PATH = "lmms-lab/VideoMMMU"

@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_videommmu_dataset(model_args, data_args, training_args, *args, **kwargs):
    subsets = []
    for subset_name in SUBSET_NAMES:
        dataset = load_dataset(DATASET_HF_PATH, subset_name, split="test")
        # Add subset column for path lookup in data_prepare
        new_column = [subset_name] * len(dataset)
        dataset = dataset.add_column("subset", new_column)
        subsets.append(dataset)
    
    dataset = datasets.concatenate_datasets(subsets)
    
    kwargs['dataset_name'] = DATASET_PARSER_NAME
    kwargs['global_dataset_name'] = DATASET_PARSER_NAME
    
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
    
    return dataset, None