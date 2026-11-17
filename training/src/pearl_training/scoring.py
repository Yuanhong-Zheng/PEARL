"""Use the released PEARL answer parser without changing the baseline."""

import importlib.util
from pathlib import Path

# utils.py of the released inference code at the repository root.
_path = Path(__file__).resolve().parents[3] / "utils.py"
_spec = importlib.util.spec_from_file_location("pearl_released_utils", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
extract_answer = _module.extract_answer_from_response


def rescore(result):
    for record in result["records"]:
        for rotation in record["rotations"]:
            rotation["predicted"] = extract_answer(rotation["response"]).strip().upper()
            rotation["correct"] = rotation["predicted"] == rotation["expected"]
        record["correct"] = all(rotation["correct"] for rotation in record["rotations"])
    result["strict_correct"] = sum(record["correct"] for record in result["records"])
    result["answer_parser"] = "released_inference/utils.py::extract_answer_from_response"
    return result
