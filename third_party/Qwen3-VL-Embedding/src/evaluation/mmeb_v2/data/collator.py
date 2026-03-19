import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MultimodalEvalDataCollator:
    encode_side: str  # 'qry' or 'cand'

    def __call__(self, examples):
        # Select input key based on encoding side
        input_key = "query_input" if self.encode_side == 'qry' else "cand_input"
        
        # Extract batch inputs: List[List[Dict]]
        batch_inputs = [ex[input_key] for ex in examples]
        
        # Extract metadata
        dataset_infos = [ex["dataset_infos"] for ex in examples]
        
        return batch_inputs, dataset_infos