import os
import cv2

from .base_eval_dataset import AutoEvalPairDataset, add_metainfo_hook
from ...constant import EVAL_DATASET_HF_PATH
from ...utils.dataset_utils import load_hf_dataset, sample_dataset
from ...utils.vision_utils.vision_utils import process_video_frames, load_frames


TASK_PROMPT = "Given a video and a question, select the most accurate answer from the provided candidates. Return only the exact text of your chosen answer."


@add_metainfo_hook
def data_prepare(batch_dict, *args, **kwargs):
    max_frames_saved = kwargs['max_frames_saved']
    video_root = kwargs['video_root']
    frame_root = kwargs['frame_root']
    num_frames = kwargs['num_frames']
    
    query_inputs, cand_inputs, dataset_infos = [], [], []
    
    for video_idx, query, answer_idx, question_idx, options in zip(
        batch_dict['video_idx'],
        batch_dict['question'],
        batch_dict['answer'],
        batch_dict['question_idx'],
        batch_dict['option']
    ):
        answer_idx = int(answer_idx)
        
        query_text = query + ' ' + ' '.join(options)
        
        video_path = f'{video_root}/{video_idx}.mp4'
        frame_dir = f'{frame_root}/{video_idx}'
        frames = load_frames(frame_dir)
        
        if not frames:
            print(f'Extracting frames for: {video_path}')
            os.makedirs(frame_dir, exist_ok=True)
            assert os.path.exists(video_path)
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // max_frames_saved)
            frame_idx = 0
            saved_frames = 0
            while saved_frames < max_frames_saved:
                assert cap.isOpened(), "not cap.isOpened()"
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(frame_dir, f"{saved_frames:04d}.jpeg")
                cv2.imwrite(frame_path, frame)
                saved_frames += 1
                frame_idx += step
            cap.release()
            print(f'[{DATASET_PARSER_NAME}] Extracted #frames: {saved_frames}, dumped to {frame_dir}')

        qry_frame_paths = process_video_frames(frame_dir, num_frames=num_frames)
        
        query_inputs.append({
            "text": query_text,
            "video": qry_frame_paths,
            "instruction": TASK_PROMPT,
        })
        
        # Extract option text, removing prefix like "A. "
        cand_texts = [o[o.find('. '):].strip('. ') for o in options]
        cand_inputs.append([{"text": t} for t in cand_texts])
        
        dataset_infos.append({
            "question_id": question_idx,
            "video_id": video_idx,
            "query": query_text,
            "cand_names": options,
            "answer": options[answer_idx],
            "label_name": options[answer_idx],
            "answer_idx": answer_idx,
            "qry_frame_paths": qry_frame_paths,
        })

    return {
        "query_input": query_inputs,
        "cand_input": cand_inputs,
        "dataset_infos": dataset_infos,
    }


DATASET_PARSER_NAME = "egoschema"


@AutoEvalPairDataset.register(DATASET_PARSER_NAME)
def load_egoschema_dataset(model_args, data_args, *args, **kwargs):
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
    corpus = None

    return dataset, corpus