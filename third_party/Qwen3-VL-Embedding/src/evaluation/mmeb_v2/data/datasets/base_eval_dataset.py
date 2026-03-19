from abc import ABCMeta, abstractmethod
from functools import wraps
from datasets import Dataset, Features, Value, Sequence

# Schema adapted for Qwen3VL's dictionary list structure
INPUT_FEATURE_DICT = {
    "text": Value(dtype='string'),
    "image": Value(dtype='string'),
    "video": Sequence(Value(dtype='string')),
    "instruction": Value(dtype='string'),
    "fps": Value(dtype='float32'),
    "max_frames": Value(dtype='int32'),
}

EVAL_QRY_FEATURES = Features({
    "query_input": INPUT_FEATURE_DICT,
    "dataset_infos": {
        "cand_names": Sequence(Value(dtype='string')),
        "label_name": Value(dtype='string')
    }
})

EVAL_CAND_FEATURES = Features({
    "cand_input": Sequence(INPUT_FEATURE_DICT),
    "dataset_infos": {
        "cand_name": Value(dtype='string'),
    },
})


class AutoEvalPairDataset(metaclass=ABCMeta):
    registry = {}

    def __init_subclass__(cls):
        if cls.__name__ not in AutoEvalPairDataset.registry:
            AutoEvalPairDataset.registry[cls.__name__] = cls
        else:
            raise RuntimeError(f'Subclass "{cls.__name__}" has already defined.')

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(...)` methods."
        )

    @classmethod
    def instantiate(cls, dataset_parser, *args, **kwargs):
        try:
            return cls.registry[dataset_parser](*args, **kwargs)
        except Exception as e:
            raise e

    @classmethod
    def register(cls, dataset_name):
        def inner_wrapper(wrapped_class):
            if dataset_name in cls.registry:
                print(f"[Alert] AutoPairDataset: a class in the same name ({dataset_name}) has been registered")
            else:
                cls.registry[dataset_name] = wrapped_class
            return wrapped_class
        return inner_wrapper

    @abstractmethod
    def main(self):
        pass


def add_metainfo_hook(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        batch_data = f(*args, **kwargs)
        sample_list = batch_data.get('query_input', batch_data.get('cand_input', []))
        batch_size = len(sample_list)
        global_dataset_name = kwargs.get("global_dataset_name", "None")
        batch_data['global_dataset_name'] = [global_dataset_name] * batch_size
        return batch_data
    return wrapper

def generate_cand_dataset(dataset, corpus):
    cand_rows = []
    all_cand_name = set()
    
    for row in dataset:
        assert len(row["cand_input"]) == len(row["dataset_infos"]["cand_names"]), \
            f"Mismatch: cand_input({len(row['cand_input'])}) vs cand_names({len(row['dataset_infos']['cand_names'])})"
        
        for cand_input, cand_name in zip(row["cand_input"], row["dataset_infos"]["cand_names"]):
            if cand_name not in all_cand_name:
                cand_rows.append({
                    "cand_input": [cand_input][0],
                    "dataset_infos": {"cand_name": cand_name},
                })
                all_cand_name.add(cand_name)

    if corpus is not None:
        for row in corpus:
            assert len(row["cand_input"]) == len(row["dataset_infos"]["cand_names"]) == 1
            
            cand_name = row["dataset_infos"]["cand_names"][0]
            if cand_name not in all_cand_name:
                cand_rows.append({
                    "cand_input": row["cand_input"][0],
                    "dataset_infos": {"cand_name": cand_name},
                })
                all_cand_name.add(cand_name)

    return Dataset.from_list(cand_rows)