"""Unit tests for eval/main_table.py on synthetic PSVU-Bench outputs.

Run from the repository root: python -m unittest discover -s tests
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import main_table  # noqa: E402

QA_TYPE = {name: "current-time qa" for name in main_table.REAL_TIME}
QA_TYPE.update({name: "past-time qa" for name in main_table.PAST_TIME})

# (correct, total) per category. With these counts the table row equals the paper's
# Qwen3-VL-8B+PEARL rows without training and with training on PSVU-10K.
UNTRAINED = {"Presence": (198, 311), "Behavior": (92, 184), "Appearance": (102, 184),
             "Location": (71, 134), "Relation": (58, 109), "Action": (174, 359),
             "Event-based": (102, 193), "Time-based": (98, 201)}
UNTRAINED_ROW = ["63.67", "50.00", "55.43", "52.99", "53.21", "48.47", "53.96",
                 "52.85", "48.76", "50.80", "52.38"]
TRAINED = {"Presence": (213, 311), "Behavior": (115, 184), "Appearance": (115, 184),
           "Location": (88, 134), "Relation": (70, 109), "Action": (188, 359),
           "Event-based": (120, 193), "Time-based": (117, 201)}
TRAINED_ROW = ["68.49", "62.50", "62.50", "65.67", "64.22", "52.37", "62.62",
               "62.18", "58.21", "60.19", "61.41"]


def write_split(root, split, counts, *, files=3, category_field=lambda name: name,
                rotation=True, write_results=True, drop_predictions=()):
    """Write annotations and inference outputs for one split; return (results, annotations)."""
    annotations = Path(root) / split / "annotations"
    results = Path(root) / split / "results"
    annotations.mkdir(parents=True)
    results.mkdir(parents=True)
    videos = [[] for _ in range(files)]
    details = [[] for _ in range(files)]
    outputs = [[] for _ in range(files)]
    index = 0
    for name, (correct, total) in counts.items():
        for position in range(total):
            slot = index % files
            qa_id = index
            index += 1
            qa = {"id": qa_id, "qa_type": QA_TYPE[name], "time": "00:01:00",
                  "question": f"Question {qa_id} about {{A}}?", "optionA": "a", "optionB": "b",
                  "optionC": "c", "optionD": "d", "gt": "A"}
            category = category_field(name)
            if category is not None:
                qa["category"] = category
            videos[slot].append(qa)
            is_correct = position < correct
            if qa_id in drop_predictions:
                continue
            details[slot].append({"id": qa_id, "qa_type": QA_TYPE[name], "gt": "A",
                                  "predicted": "A" if is_correct else "", "is_correct": is_correct})
            flags = [True, True, True, is_correct]
            outputs[slot].append({**qa, "rotation_enabled": rotation,
                                  "rotation_correct_flags": flags if rotation else []})
    for slot in range(files):
        stem = f"{split}_video{slot}"
        definition = {"id": 10_000 + slot, "qa_type": "concept definition", "time": "00:00:10",
                      "question": "This is {A}."}
        (annotations / f"{stem}.json").write_text(json.dumps(
            [{"video_path": f"videos/{stem}.mp4", "timestamps": [definition] + videos[slot]}]))
        (results / f"{stem}_evaluation.json").write_text(json.dumps({"details": details[slot]}))
        if write_results:
            (results / f"{stem}_result.json").write_text(json.dumps(
                [{"video_path": f"videos/{stem}.mp4", "timestamps": outputs[slot]}]))
    return str(results), str(annotations)


def split_counts(counts, names):
    return {name: counts[name] for name in names}


ENTITY = [name for name in main_table.CATEGORIES if name != "Action"]


class MainTableTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = self.directory.name

    def tearDown(self):
        self.directory.cleanup()

    def paper_runs(self, counts):
        return [write_split(self.root, "frame-level", split_counts(counts, ENTITY)),
                write_split(self.root, "video-level", split_counts(counts, ["Action"]), files=2)]

    def test_reproduces_paper_rows(self):
        report = main_table.score(self.paper_runs(UNTRAINED))
        self.assertEqual(main_table.row_values(report), UNTRAINED_ROW)
        self.assertEqual(report["missing_categories"], [])
        self.assertEqual(report["issues"], {})
        self.assertEqual(report["protocol_counts"], {"four_rotations": 1675})

    def test_reproduces_trained_row_through_cli(self):
        runs = self.paper_runs(TRAINED)
        output = Path(self.root) / "report.json"
        argv = ["--name", "Qwen3-VL-8B+PEARL", "--latex", "--json", str(output), "--fail-on-issues"]
        for result_dir, annotation_dir in runs:
            argv += ["--run", result_dir, annotation_dir]
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            status = main_table.main(argv)
        self.assertEqual(status, 0)
        self.assertIn("| Qwen3-VL-8B+PEARL | " + " | ".join(TRAINED_ROW) + " |", stdout.getvalue())
        self.assertIn("Qwen3-VL-8B+PEARL & " + " & ".join(TRAINED_ROW) + r" \\", stdout.getvalue())
        saved = json.loads(output.read_text())
        self.assertEqual(saved["categories"]["Action"], {"correct": 188, "total": 359,
                                                         "accuracy": 100 * 188 / 359})

    def test_missing_category_is_reported(self):
        runs = [write_split(self.root, "frame-level", split_counts(UNTRAINED, ENTITY))]
        report = main_table.score(runs)
        self.assertEqual(report["missing_categories"], ["Action"])
        self.assertFalse(report["real_time_complete"])
        self.assertTrue(report["past_time_complete"])
        values = main_table.row_values(report)
        self.assertEqual(values[5], "--")
        expected = sum(100 * c / t for name, (c, t) in UNTRAINED.items() if name in main_table.REAL_TIME[:5]) / 5
        self.assertEqual(values[6], f"{expected:.2f}*")
        self.assertTrue(values[10].endswith("*"))
        self.assertTrue(any("Action" in line for line in main_table.notes(report)))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main_table.main(["--run", *runs[0], "--fail-on-issues"]), 1)

    def test_no_past_time_questions(self):
        runs = [write_split(self.root, "video-level", split_counts(UNTRAINED, ["Action"]))]
        report = main_table.score(runs)
        self.assertEqual(report["categories"]["Action"]["total"], 359)
        self.assertIsNone(report["past_time_avg"])
        self.assertIsNone(report["overall_avg"])
        self.assertEqual(main_table.row_values(report)[-1], "--")

    def test_aliases_unlabeled_and_unknown_categories(self):
        names = {"Event-based": "Event", "Time-based": "time_based", "Presence": "presence",
                 "Behavior": None, "Appearance": "Clothing"}
        counts = {name: (1, 2) for name in names}
        runs = [write_split(self.root, "frame-level", counts, files=1,
                            category_field=lambda name: names[name])]
        report = main_table.score(runs)
        self.assertEqual(report["categories"]["Event-based"]["total"], 2)
        self.assertEqual(report["categories"]["Time-based"]["total"], 2)
        self.assertEqual(report["categories"]["Presence"]["total"], 2)
        self.assertEqual(len(report["issues"]["uncategorized"]), 2)
        self.assertEqual(len(report["issues"]["invalid_category"]), 2)
        self.assertIn("'Clothing'", report["issues"]["invalid_category"][0])
        self.assertEqual(report["categories"]["Behavior"]["total"], 0)

    def test_category_conflicting_with_qa_type(self):
        counts = {"Presence": (1, 1)}
        runs = [write_split(self.root, "frame-level", counts, files=1,
                            category_field=lambda name: "Event-based")]
        report = main_table.score(runs)
        self.assertEqual(report["categories"]["Event-based"]["total"], 1)
        self.assertEqual(len(report["issues"]["category_qa_type_conflict"]), 1)

    def test_missing_predictions(self):
        counts = {"Presence": (2, 4), "Event-based": (1, 2)}
        runs = [write_split(self.root, "frame-level", counts, files=1, drop_predictions={0})]
        wrong = main_table.score(runs)
        self.assertEqual(wrong["categories"]["Presence"], {"correct": 1, "total": 4, "accuracy": 25.0})
        skip = main_table.score(runs, missing="skip")
        self.assertEqual(skip["categories"]["Presence"]["total"], 3)
        self.assertEqual(len(skip["issues"]["missing_prediction"]), 1)

    def test_rotation_checks(self):
        counts = {"Presence": (1, 2), "Event-based": (1, 2)}
        runs = [write_split(self.root, "frame-level", counts, files=1, rotation=False)]
        report = main_table.score(runs)
        self.assertEqual(len(report["issues"]["single_pass"]), 4)

        results, _ = runs[0]
        result_file = Path(results) / "frame-level_video0_result.json"
        data = json.loads(result_file.read_text())
        for qa in data[0]["timestamps"]:
            qa["rotation_enabled"] = True
            qa["rotation_correct_flags"] = [True, True, True, True]
        result_file.write_text(json.dumps(data))
        report = main_table.score(runs)
        self.assertEqual(len(report["issues"]["rotation_flag_mismatch"]), 2)

        result_file.unlink()
        report = main_table.score(runs)
        self.assertEqual(report["protocol_counts"], {"unchecked": 4})
        self.assertEqual(report["issues"], {})

    def test_strict_correct(self):
        self.assertTrue(main_table.strict_correct([True] * 4))
        self.assertFalse(main_table.strict_correct([True, True, True, False]))
        self.assertFalse(main_table.strict_correct([True] * 3))
        self.assertFalse(main_table.strict_correct([]))

    def test_unmatched_prediction(self):
        runs = [write_split(self.root, "frame-level", {"Presence": (1, 1)}, files=1)]
        results, _ = runs[0]
        extra = Path(results) / "other_evaluation.json"
        extra.write_text(json.dumps({"details": [{"id": 1, "qa_type": "past-time qa", "is_correct": True}]}))
        report = main_table.score(runs)
        self.assertEqual(len(report["issues"]["unmatched_prediction"]), 1)


if __name__ == "__main__":
    unittest.main()
