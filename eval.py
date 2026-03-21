#!/usr/bin/env python3
"""
Compute average accuracy across all evaluation.json files.
Supports specifying ignored questions through a JSON file.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_ignore_list(ignore_file_path):
    """
    Load the list of questions to ignore.

    Args:
        ignore_file_path: Path to the ignore-list file

    Returns:
        dict: {video_name: set(qa_ids)}

    Example JSON format:
    {
        "ignored_questions": [
            {"video": "xiaohei", "qa_id": 14},
            {"video": "juexing", "qa_id": 10}
        ]
    }
    Or:
    [
        {"video": "xiaohei", "qa_id": 14},
        {"video": "juexing", "qa_id": 10}
    ]
    """
    if not ignore_file_path or not Path(ignore_file_path).exists():
        return {}

    try:
        with open(ignore_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON formats
        if isinstance(data, dict) and "ignored_questions" in data:
            questions = data["ignored_questions"]
        elif isinstance(data, list):
            questions = data
        else:
            print("⚠ Warning: invalid ignore-file format; expected a list or a dict containing 'ignored_questions'")
            return {}

        # Build a mapping from video_name -> set(qa_ids)
        ignore_dict = defaultdict(set)
        for item in questions:
            video = item.get("video", "")
            qa_id = item.get("qa_id")
            if video and qa_id is not None:
                ignore_dict[video].add(qa_id)

        return ignore_dict

    except Exception as e:
        print(f"⚠ Warning: failed to read ignore file: {e}")
        return {}


def should_ignore_question(video_name, qa_id, ignore_dict):
    """
    Determine whether a question should be ignored.

    Args:
        video_name: Video name without the _evaluation suffix
        qa_id: Question ID
        ignore_dict: Ignore mapping
        
    Returns:
        bool: True if the question should be ignored
    """
    return qa_id in ignore_dict.get(video_name, set())


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute average accuracy across all evaluation.json files")
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=DEFAULT_RESULT_DIR,
        help="Directory containing evaluation.json and result.json files (positional argument)",
    )
    parser.add_argument(
        "--result_dir",
        dest="result_dir_flag",
        type=str,
        default=None,
        help="Directory containing evaluation.json and result.json files (backward-compatible flag)",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        default=DEFAULT_IGNORE_FILE,
        help="Path to the ignore-list JSON file (contains questions to ignore)",
    )
    args = parser.parse_args()

    # Prefer --result_dir over the positional argument
    result_dir = args.result_dir_flag if args.result_dir_flag else args.result_dir

    # Resolve the result directory
    script_dir = Path(result_dir)

    if not script_dir.exists():
        print(f"Error: specified directory does not exist: {script_dir}")
        return

    print(f"Result directory: {script_dir}\n")

    # Load the ignore list
    ignore_dict = load_ignore_list(args.ignore)
    if ignore_dict:
        total_ignored = sum(len(ids) for ids in ignore_dict.values())
        print(f"✓ Ignore list loaded; {total_ignored} question(s) will be ignored")
        print(f"  Spanning {len(ignore_dict)} video(s)\n")
    else:
        print("No ignore list used\n")

    # Initialize counters
    total_count = 0
    correct_count = 0
    current_time_count = 0
    current_time_correct = 0
    past_time_count = 0
    past_time_correct = 0

    # Ignore statistics
    total_ignored = 0
    ignored_by_type = {"current-time qa": 0, "past-time qa": 0}

    # Store per-file statistics
    file_stats = []

    # Iterate over all evaluation.json files
    evaluation_files = sorted(script_dir.glob("*_evaluation.json"))

    if not evaluation_files:
        print("Error: no evaluation.json files were found")
        return

    print(f"Found {len(evaluation_files)} evaluation file(s)\n")
    print("=" * 80)

    for eval_file in evaluation_files:
        try:
            # Load evaluation.json
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            # Extract the video name
            video_name = eval_file.stem.replace("_evaluation", "")

            # Use the per-question details from evaluation.json for unified statistics
            details = eval_data.get("details", [])

            file_total = 0
            file_correct = 0
            file_current_count = 0
            file_current_correct = 0
            file_past_count = 0
            file_past_correct = 0
            file_ignored = 0

            for item in details:
                qa_id = item.get("id")
                qa_type = item.get("qa_type", "")

                # Check whether this question should be ignored
                if should_ignore_question(video_name, qa_id, ignore_dict):
                    file_ignored += 1
                    if qa_type in ignored_by_type:
                        ignored_by_type[qa_type] += 1
                    continue

                # Only count current-time qa and past-time qa
                if qa_type not in ["current-time qa", "past-time qa"]:
                    continue

                is_correct = item.get("is_correct", False)

                file_total += 1
                if is_correct:
                    file_correct += 1

                if qa_type == "current-time qa":
                    file_current_count += 1
                    if is_correct:
                        file_current_correct += 1
                elif qa_type == "past-time qa":
                    file_past_count += 1
                    if is_correct:
                        file_past_correct += 1

            total_ignored += file_ignored

            # Accumulate statistics
            total_count += file_total
            correct_count += file_correct
            current_time_count += file_current_count
            current_time_correct += file_current_correct
            past_time_count += file_past_count
            past_time_correct += file_past_correct

            # Compute accuracies
            file_total_accuracy = file_correct / file_total if file_total > 0 else 0
            file_current_accuracy = file_current_correct / file_current_count if file_current_count > 0 else 0
            file_past_accuracy = file_past_correct / file_past_count if file_past_count > 0 else 0

            # Save per-file statistics
            file_stats.append(
                {
                    "name": eval_file.name,
                    "total_accuracy": file_total_accuracy,
                    "total_count": file_total,
                    "correct_count": file_correct,
                    "current_time_accuracy": file_current_accuracy,
                    "current_time_count": file_current_count,
                    "current_time_correct": file_current_correct,
                    "past_time_accuracy": file_past_accuracy,
                    "past_time_count": file_past_count,
                    "past_time_correct": file_past_correct,
                    "ignored_count": file_ignored,
                }
            )

            # Print per-file statistics
            print(f"File: {eval_file.name}")
            print(f"  Overall: {file_correct}/{file_total} = {file_total_accuracy:.2%}")
            print(f"  Current-time: {file_current_correct}/{file_current_count} = {file_current_accuracy:.2%}")
            print(f"  Past-time: {file_past_correct}/{file_past_count} = {file_past_accuracy:.2%}")
            if file_ignored > 0:
                print(f"  Ignored questions: {file_ignored}")
            print()

        except Exception as e:
            print(f"Warning: error reading file {eval_file.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    current_time_accuracy = current_time_correct / current_time_count if current_time_count > 0 else 0
    past_time_accuracy = past_time_correct / past_time_count if past_time_count > 0 else 0
    avg_accuracy = (current_time_accuracy + past_time_accuracy) / 2

    # Print overall statistics
    print("=" * 80)
    print("\nOverall Statistics\n")
    print("=" * 80)

    if total_ignored > 0:
        print("\nIgnore Statistics:")
        print(f"  Total ignored questions: {total_ignored}")
        print(f"  Current-time qa ignored: {ignored_by_type.get('current-time qa', 0)}")
        print(f"  Past-time qa ignored: {ignored_by_type.get('past-time qa', 0)}")

    print("\nAverage Accuracy:")
    print(f"  Accuracy: {avg_accuracy:.2%}")

    print("\nCurrent-time QA Accuracy:")
    print(f"  Correct answers: {current_time_correct}")
    print(f"  Total questions: {current_time_count}")
    print(f"  Accuracy: {current_time_accuracy:.2%} ({current_time_correct}/{current_time_count})")

    print("\nPast-time QA Accuracy:")
    print(f"  Correct answers: {past_time_correct}")
    print(f"  Total questions: {past_time_count}")
    print(f"  Accuracy: {past_time_accuracy:.2%} ({past_time_correct}/{past_time_count})")

    print("\n" + "=" * 80)

    # Save overall statistics to a file
    output_file = script_dir / "overall_statistics.json"
    overall_stats = {
        "total_files": len(evaluation_files),
        "ignored_info": {
            "total_ignored": total_ignored,
            "current_time_ignored": ignored_by_type.get("current-time qa", 0),
            "past_time_ignored": ignored_by_type.get("past-time qa", 0),
            "ignore_file_used": args.ignore if args.ignore else None,
        },
        "avg": {
            "accuracy": avg_accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
        },
        "current_time": {
            "accuracy": current_time_accuracy,
            "correct_count": current_time_correct,
            "total_count": current_time_count,
        },
        "past_time": {
            "accuracy": past_time_accuracy,
            "correct_count": past_time_correct,
            "total_count": past_time_count,
        },
        "per_file_stats": file_stats,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(overall_stats, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Statistics saved to: {output_file.name}")


if __name__ == "__main__":
    # ============ Update Default Parameters Here ============
    DEFAULT_RESULT_DIR = "output_results/frame-level/qwen3vl_k4_n1_fps1"
    DEFAULT_IGNORE_FILE = None
    # ==========================================
    main()
