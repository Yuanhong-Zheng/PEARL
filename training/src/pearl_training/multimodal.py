from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info


IMAGE_PIXELS = 256 * 32 * 32
# Answering-VLM (reader) prompt used for training and validation. It reproduces the
# released evaluation prompt (video_qa_inference.py::build_messages together with
# utils.py::build_question_with_options).
READER_INPUT_PROTOCOL = "pearl-eval-v1"
# Protocol of the teacher verification prompts used to build PSVU-10K (they use
# stream_clock_text). Data-construction and verification receipts carry this value,
# so it stays unchanged and is checked when samples are loaded for training.
VERIFIER_INPUT_PROTOCOL = "causal-stream-v2"
QA_INSTRUCTION = "Please output your answer choice in <ans></ans> tags."


def stream_clock_text(sample):
    """Only client-observable timing; annotation-window/evidence times stay hidden."""
    clip = sample["current_clip"]
    return (f"The question is asked at source time {sample['question_time']:.3f} seconds "
            f"in one continuous video stream starting at {sample.get('visible_start', 0):.3f} seconds. "
            f"The CURRENT clip covers source [{clip['start']:.3f}, {clip['end']:.3f}) seconds. "
            "Historical clip times below refer to that same stream. Gaps are unprovided history, "
            "not a reset of the stream. Do not interpret the first retrieved clip as the beginning "
            "of the video. Use only the supplied pixels and runtime concept definitions.")


def image_part(path):
    return {"type": "image", "image": path, "min_pixels": 4 * 32 * 32, "max_pixels": IMAGE_PIXELS}


def video_part(clip):
    return {"type": "video", "video": clip["frames"], "sample_fps": clip["sample_fps"],
            "min_pixels": 4 * 32 * 32, "max_pixels": IMAGE_PIXELS}


def concept_parts(sample):
    parts = []
    for concept in sample["concepts"]:
        parts.extend([image_part(concept["image"]), {"type": "text", "text": "Runtime concept {" + concept["name"] + "}: " + concept["description"]}])
    return parts


def concept_definition_text(concept):
    """PSVU-Bench-style concept definition sentence built from the stored description."""
    description = concept["description"]
    return f"{description[:1].upper()}{description[1:]} is named {{{concept['name']}}}. Please remember this name."


def reader_concept_parts(sample):
    """Reference image of each concept, followed by its definition text (evaluation format)."""
    parts = []
    for concept in sample["concepts"]:
        parts.extend([image_part(concept["image"]), {"type": "text", "text": concept_definition_text(concept)}])
    return parts


def question_text(sample, rotation=0):
    options = sample["options"][rotation:] + sample["options"][:rotation]
    text = sample["question"] + "\n" + "\n".join(f"{letter}. {option}" for letter, option in zip("ABCD", options))
    return text + "\n\n" + QA_INSTRUCTION


def answer_text(sample, rotation=0):
    return "<ans>" + "ABCD"[(sample["answer_index"] - rotation) % 4] + "</ans>"


def sft_messages(sample, rotation=0, context="oracle_noise", input_protocol=READER_INPUT_PROTOCOL):
    positive = set(sample.get("context_positive_clip_ids", sample["positive_clip_ids"]))
    negative = set(sample["negative_clip_ids"][:2]) if context == "oracle_noise" else set()
    if context not in {"oracle", "oracle_noise"}:
        raise ValueError("Unsupported training context; implement and verify retrieval before using it")
    history = [clip for clip in sample["candidates"] if clip["id"] in positive | negative]
    return qa_messages(sample, history, rotation, input_protocol=input_protocol)


def qa_messages(sample, history, rotation=0, input_protocol=READER_INPUT_PROTOCOL):
    """Build the released evaluation prompt from explicitly selected, causal history.

    Same order and texts as video_qa_inference.py::build_messages: concept references
    with their definitions, the history clips (no per-clip timestamps), the current
    clip, then the question, its options and the answer-format instruction.
    """
    if any(clip["end"] > sample["current_clip"]["start"] for clip in history):
        raise ValueError("Reader history contains current or future pixels")
    if input_protocol != READER_INPUT_PROTOCOL:
        raise ValueError("Unknown reader input protocol")
    parts = reader_concept_parts(sample)
    if history:
        parts.append({"type": "text", "text": "Here are the previous related video clips:"})
        parts.extend(video_part(clip) for clip in history)
    parts.extend([{"type": "text", "text": "The following is the current video clip:"}, video_part(sample["current_clip"])])
    parts.append({"type": "text", "text": question_text(sample, rotation)})
    return [{"role": "user", "content": parts}]


def query_messages(sample):
    # The answer options and evidence labels are intentionally not in a query.
    return [
        {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": concept_parts(sample) + [{"type": "text", "text": sample["question"]}]},
    ]


def document_messages(clip):
    return [
        {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": [video_part(clip)]},
    ]


def process_messages(processor, messages, max_length=8192, assistant=None):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, video_inputs, kwargs = process_vision_info(messages, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
    videos, metadata = zip(*video_inputs) if video_inputs else (None, None)
    common = dict(images=images, videos=list(videos) if videos else None,
                  video_metadata=list(metadata) if metadata else None,
                  do_resize=False, return_tensors="pt", **kwargs)
    prefix = processor(text=[text], **common)
    if assistant is None:
        inputs = prefix
    else:
        full_text = processor.apply_chat_template(messages + [{"role": "assistant", "content": assistant}], tokenize=False, add_generation_prompt=False)
        inputs = processor(text=[full_text], **common)
        length = prefix["input_ids"].shape[1]
        if not torch.equal(inputs["input_ids"][:, :length], prefix["input_ids"]):
            raise ValueError("Assistant prefix changed during tokenization")
        labels = inputs["input_ids"].clone()
        labels[:, :length] = -100
        if not torch.any(labels != -100):
            raise ValueError("No assistant supervision tokens")
        inputs["labels"] = labels
    if inputs["input_ids"].shape[1] > max_length:
        raise ValueError(f"Input uses {inputs['input_ids'].shape[1]} tokens, exceeding {max_length}; evidence was not truncated")
    return inputs


def pool_embedding(model, inputs):
    outputs = model(**inputs)
    mask = inputs["attention_mask"]
    columns = mask.shape[1] - 1 - mask.flip(1).argmax(1)
    rows = torch.arange(mask.shape[0], device=mask.device)
    return F.normalize(outputs.last_hidden_state[rows, columns].float(), dim=-1)


def embedding_model_class():
    # Released model file under the repository root.
    path = Path(__file__).resolve().parents[3] / "third_party/Qwen3-VL-Embedding/src/models/qwen3_vl_embedding.py"
    spec = importlib.util.spec_from_file_location("pearl_frozen_qwen_embedding", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Qwen3VLForEmbedding


def lora_targets(model):
    targets = [name for name, module in model.named_modules() if ".language_model." in name and name.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"} and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("Could not identify language-only LoRA targets")
    return targets
