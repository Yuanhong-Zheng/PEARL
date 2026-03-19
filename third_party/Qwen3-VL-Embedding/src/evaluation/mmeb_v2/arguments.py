from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "huggingface model name or path"}
    )
    normalize: bool = field(
        default=True, 
        metadata={"help": "normalize query and passage representations"}
    )
    instruction: str = field(
        default="Represent the user's input.", 
        metadata={"help": "default instruction for the model"}
    )
    device: str = field(
        default="cuda", 
        metadata={"help": "use cuda for single GPU inference, if multiple GPUs are available it will use DP automatically"}
    )

@dataclass
class DataArguments:
    dataset_config: str = field(
        default=None, 
        metadata={"help": "yaml file with dataset configuration"}
    )
    data_basedir: str = field(
        default=None, 
        metadata={"help": "Expect an absolute path to the base directory of all datasets. If set, it will be prepended to each dataset path"}
    )
    encode_output_path: str = field(
        default=None, 
        metadata={"help": "encode output path"}
    )

@dataclass
class EvalArguments(TrainingArguments):
    pass