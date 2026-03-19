#!/usr/bin/env python3
"""
检测 annotations_short 中可向后平移固定时长、且仍位于同一 clip 的 QA 条目。

默认只检查：
- current-time qa
- past-time qa

判定规则与 video_qa_inference.py 保持一致：
- 使用 start_time <= t < end_time 判断时间点所在 clip
- 若 t 和 t + delta 落在同一个 clip，则认为该条目标注时间可安全后移 delta

本脚本只做检测，不修改原始数据集。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import time_to_seconds


DEFAULT_QA_TYPES = ("current-time qa", "past-time qa")


def format_hhmmss(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) / 1000
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def get_clip_at_time(clips: List[Dict], time_seconds: float) -> Optional[Dict]:
    """复用当前推理逻辑：start_time <= t < end_time。"""
    for clip in clips:
        if clip["start_time"] <= time_seconds < clip["end_time"]:
            return clip
    return None


def load_clips(clips_info_path: Path) -> List[Dict]:
    data = json.loads(clips_info_path.read_text(encoding="utf-8"))
    clips = data.get("clips", [])
    if not isinstance(clips, list):
        raise ValueError(f"Invalid clips format in {clips_info_path}")
    return clips


def iter_annotation_entries(annotation_path: Path):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        records = data
    else:
        records = [data]

    for record_idx, record in enumerate(records):
        timestamps = record.get("timestamps", [])
        for qa_idx, qa_item in enumerate(timestamps):
            yield record_idx, qa_idx, record, qa_item


def detect_shiftable_entries(
    annotations_dir: Path,
    clips_root: Path,
    delta_seconds: float,
    qa_types: List[str],
) -> Dict:
    qa_type_set = set(qa_types)
    results = []
    skipped_missing_clips_info = []
    skipped_time_not_found = []

    annotation_files = sorted(annotations_dir.glob("*.json"))
    checked_annotations = 0
    checked_qas = 0

    for annotation_path in annotation_files:
        annotation_data = json.loads(annotation_path.read_text(encoding="utf-8"))
        records = annotation_data if isinstance(annotation_data, list) else [annotation_data]
        if not records:
            continue

        first_video_path = records[0].get("video_path", "")
        video_name = Path(first_video_path).stem if first_video_path else annotation_path.stem
        clips_info_path = clips_root / video_name / f"{video_name}_clips_info.json"
        if not clips_info_path.exists():
            skipped_missing_clips_info.append(str(annotation_path))
            continue

        clips = load_clips(clips_info_path)
        checked_annotations += 1

        for record_idx, qa_idx, record, qa_item in iter_annotation_entries(annotation_path):
            qa_type = qa_item.get("qa_type")
            if qa_type not in qa_type_set:
                continue

            checked_qas += 1
            time_str = qa_item.get("time")
            if not time_str:
                skipped_time_not_found.append({
                    "annotation_path": str(annotation_path),
                    "record_idx": record_idx,
                    "qa_idx": qa_idx,
                    "id": qa_item.get("id"),
                    "reason": "missing time",
                })
                continue

            time_seconds = time_to_seconds(time_str)
            shifted_time_seconds = time_seconds + delta_seconds

            current_clip = get_clip_at_time(clips, time_seconds)
            shifted_clip = get_clip_at_time(clips, shifted_time_seconds)

            if current_clip is None or shifted_clip is None:
                skipped_time_not_found.append({
                    "annotation_path": str(annotation_path),
                    "record_idx": record_idx,
                    "qa_idx": qa_idx,
                    "id": qa_item.get("id"),
                "qa_type": qa_type,
                "time": time_str,
                "shifted_time": format_hhmmss(shifted_time_seconds),
                "reason": "time not covered by any clip",
            })
                continue

            same_clip = current_clip["clip_id"] == shifted_clip["clip_id"]
            if not same_clip:
                continue

            results.append({
                "annotation_path": str(annotation_path),
                "video_name": video_name,
                "record_idx": record_idx,
                "qa_idx": qa_idx,
                "id": qa_item.get("id"),
                "qa_type": qa_type,
                "question": qa_item.get("question", ""),
                "original_time": time_str,
                "original_time_seconds": time_seconds,
                "shifted_time": format_hhmmss(shifted_time_seconds),
                "shifted_time_seconds": shifted_time_seconds,
                "delta_seconds": delta_seconds,
                "clip_id": current_clip["clip_id"],
                "clip_start_time": current_clip["start_time"],
                "clip_end_time": current_clip["end_time"],
                "remaining_time_in_clip": round(current_clip["end_time"] - time_seconds, 3),
                "clip_path": current_clip.get("clip_path"),
            })

    summary = {
        "annotations_dir": str(annotations_dir),
        "clips_root": str(clips_root),
        "delta_seconds": delta_seconds,
        "qa_types": list(qa_types),
        "checked_annotation_files": checked_annotations,
        "checked_qas": checked_qas,
        "shiftable_qas": len(results),
        "skipped_missing_clips_info": len(skipped_missing_clips_info),
        "skipped_time_not_found": len(skipped_time_not_found),
    }

    return {
        "summary": summary,
        "results": results,
        "skipped_missing_clips_info": skipped_missing_clips_info,
        "skipped_time_not_found": skipped_time_not_found,
    }


def print_report(report: Dict, show_examples: int) -> None:
    summary = report["summary"]
    print("Detection Summary")
    print(f"  annotations_dir: {summary['annotations_dir']}")
    print(f"  clips_root: {summary['clips_root']}")
    print(f"  delta_seconds: {summary['delta_seconds']}")
    print(f"  qa_types: {', '.join(summary['qa_types'])}")
    print(f"  checked_annotation_files: {summary['checked_annotation_files']}")
    print(f"  checked_qas: {summary['checked_qas']}")
    print(f"  shiftable_qas: {summary['shiftable_qas']}")
    print(f"  skipped_missing_clips_info: {summary['skipped_missing_clips_info']}")
    print(f"  skipped_time_not_found: {summary['skipped_time_not_found']}")

    results = report["results"]
    if not results:
        print("\nNo shiftable QA entries found.")
        return

    print(f"\nFirst {min(show_examples, len(results))} shiftable entries:")
    for item in results[:show_examples]:
        print(
            f"- {item['video_name']} | id={item['id']} | {item['qa_type']} | "
            f"{item['original_time']} -> {item['shifted_time']} | "
            f"clip_id={item['clip_id']} | "
            f"clip=[{item['clip_start_time']:.2f}, {item['clip_end_time']:.2f}) | "
            f"remain={item['remaining_time_in_clip']:.3f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect QA times that can be shifted forward within the same clip.")
    parser.add_argument(
        "--annotations-dir",
        default="data/frame-level/annotations_short",
        help="Path to annotations_short directory.",
    )
    parser.add_argument(
        "--clips-root",
        default="data/frame-level/output_clips",
        help="Path to output_clips root directory.",
    )
    parser.add_argument(
        "--delta-seconds",
        type=float,
        default=0.5,
        help="Forward shift duration in seconds.",
    )
    parser.add_argument(
        "--qa-types",
        nargs="+",
        default=list(DEFAULT_QA_TYPES),
        help="QA types to inspect.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save full detection report as JSON.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=20,
        help="How many matched entries to print in stdout.",
    )
    args = parser.parse_args()

    report = detect_shiftable_entries(
        annotations_dir=Path(args.annotations_dir),
        clips_root=Path(args.clips_root),
        delta_seconds=args.delta_seconds,
        qa_types=args.qa_types,
    )
    print_report(report, show_examples=args.show_examples)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved full report to: {output_path}")


if __name__ == "__main__":
    main()
