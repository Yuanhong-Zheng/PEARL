#!/usr/bin/env python3
"""
统计所有 evaluation.json 文件的整体准确率
支持通过 JSON 文件指定需要忽略的问题
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_ignore_list(ignore_file_path):
    """
    加载需要忽略的问题列表

    Args:
        ignore_file_path: 忽略列表文件路径

    Returns:
        dict: {video_name: set(qa_ids)}

    示例 JSON 格式:
    {
        "ignored_questions": [
            {"video": "xiaohei", "qa_id": 14},
            {"video": "juexing", "qa_id": 10}
        ]
    }
    或者:
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

        # 处理不同的 JSON 格式
        if isinstance(data, dict) and "ignored_questions" in data:
            questions = data["ignored_questions"]
        elif isinstance(data, list):
            questions = data
        else:
            print("⚠ 警告: 忽略文件格式不正确，应该是列表或包含 'ignored_questions' 的字典")
            return {}

        # 构建 video_name -> set(qa_ids) 的映射
        ignore_dict = defaultdict(set)
        for item in questions:
            video = item.get("video", "")
            qa_id = item.get("qa_id")
            if video and qa_id is not None:
                ignore_dict[video].add(qa_id)

        return ignore_dict

    except Exception as e:
        print(f"⚠ 警告: 读取忽略文件失败: {e}")
        return {}


def should_ignore_question(video_name, qa_id, ignore_dict):
    """
    判断是否应该忽略某个问题

    Args:
        video_name: 视频名称（不含 _evaluation 后缀）
        qa_id: 问题 ID
        ignore_dict: 忽略字典

    Returns:
        bool: True 表示应该忽略
    """
    return qa_id in ignore_dict.get(video_name, set())


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="统计所有 evaluation.json 文件的整体准确率")
    parser.add_argument(
        "result_dir",
        nargs="?",
        default=DEFAULT_RESULT_DIR,
        help="包含 evaluation.json 和 result.json 文件的目录路径（位置参数）",
    )
    parser.add_argument(
        "--result_dir",
        dest="result_dir_flag",
        type=str,
        default=None,
        help="包含 evaluation.json 和 result.json 文件的目录路径（兼容旧参数）",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        default=DEFAULT_IGNORE_FILE,
        help="忽略列表 JSON 文件路径（包含需要忽略的问题）",
    )
    args = parser.parse_args()

    # 优先使用 --result_dir，其次使用位置参数
    result_dir = args.result_dir_flag if args.result_dir_flag else args.result_dir

    # 获取结果文件目录
    script_dir = Path(result_dir)

    if not script_dir.exists():
        print(f"错误：指定的目录不存在: {script_dir}")
        return

    print(f"统计目录: {script_dir}\n")

    # 加载忽略列表
    ignore_dict = load_ignore_list(args.ignore)
    if ignore_dict:
        total_ignored = sum(len(ids) for ids in ignore_dict.values())
        print(f"✓ 已加载忽略列表，共 {total_ignored} 个问题将被忽略")
        print(f"  涉及 {len(ignore_dict)} 个视频\n")
    else:
        print("未使用忽略列表\n")

    # 初始化统计变量
    total_count = 0
    correct_count = 0
    current_time_count = 0
    current_time_correct = 0
    past_time_count = 0
    past_time_correct = 0

    # 忽略统计
    total_ignored = 0
    ignored_by_type = {"current-time qa": 0, "past-time qa": 0}

    # 存储每个文件的统计信息
    file_stats = []

    # 遍历所有 evaluation.json 文件
    evaluation_files = sorted(script_dir.glob("*_evaluation.json"))

    if not evaluation_files:
        print("错误：没有找到任何 evaluation.json 文件")
        return

    print(f"找到 {len(evaluation_files)} 个评估文件\n")
    print("=" * 80)

    for eval_file in evaluation_files:
        try:
            # 读取 evaluation.json
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            # 获取视频名称
            video_name = eval_file.stem.replace("_evaluation", "")

            # 统一使用 evaluation.json 中的 details 逐题统计
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

                # 检查是否应该忽略
                if should_ignore_question(video_name, qa_id, ignore_dict):
                    file_ignored += 1
                    if qa_type in ignored_by_type:
                        ignored_by_type[qa_type] += 1
                    continue

                # 只统计 current-time qa 和 past-time qa
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

            # 累加统计数据
            total_count += file_total
            correct_count += file_correct
            current_time_count += file_current_count
            current_time_correct += file_current_correct
            past_time_count += file_past_count
            past_time_correct += file_past_correct

            # 计算准确率
            file_total_accuracy = file_correct / file_total if file_total > 0 else 0
            file_current_accuracy = file_current_correct / file_current_count if file_current_count > 0 else 0
            file_past_accuracy = file_past_correct / file_past_count if file_past_count > 0 else 0

            # 保存文件统计信息
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

            # 打印单个文件的统计
            print(f"文件: {eval_file.name}")
            print(f"  总体: {file_correct}/{file_total} = {file_total_accuracy:.2%}")
            print(f"  Current-time: {file_current_correct}/{file_current_count} = {file_current_accuracy:.2%}")
            print(f"  Past-time: {file_past_correct}/{file_past_count} = {file_past_accuracy:.2%}")
            if file_ignored > 0:
                print(f"  忽略问题数: {file_ignored}")
            print()

        except Exception as e:
            print(f"警告：读取文件 {eval_file.name} 时出错: {e}")
            import traceback

            traceback.print_exc()
            continue

    # 计算整体准确率
    overall_accuracy = correct_count / total_count if total_count > 0 else 0
    current_time_accuracy = current_time_correct / current_time_count if current_time_count > 0 else 0
    past_time_accuracy = past_time_correct / past_time_count if past_time_count > 0 else 0

    # 打印整体统计结果
    print("=" * 80)
    print("\n📊 整体统计结果\n")
    print("=" * 80)

    if total_ignored > 0:
        print("\n忽略统计:")
        print(f"  总共忽略问题数: {total_ignored}")
        print(f"  Current-time qa 忽略: {ignored_by_type.get('current-time qa', 0)}")
        print(f"  Past-time qa 忽略: {ignored_by_type.get('past-time qa', 0)}")

    print("\n总体准确率:")
    print(f"  正确数量: {correct_count}")
    print(f"  题目总数: {total_count}")
    print(f"  准确率: {overall_accuracy:.2%} ({correct_count}/{total_count})")

    print("\nCurrent-time QA 准确率:")
    print(f"  正确数量: {current_time_correct}")
    print(f"  题目总数: {current_time_count}")
    print(f"  准确率: {current_time_accuracy:.2%} ({current_time_correct}/{current_time_count})")

    print("\nPast-time QA 准确率:")
    print(f"  正确数量: {past_time_correct}")
    print(f"  题目总数: {past_time_count}")
    print(f"  准确率: {past_time_accuracy:.2%} ({past_time_correct}/{past_time_count})")

    print("\n" + "=" * 80)

    # 保存整体统计结果到文件
    output_file = script_dir / "overall_statistics.json"
    overall_stats = {
        "total_files": len(evaluation_files),
        "ignored_info": {
            "total_ignored": total_ignored,
            "current_time_ignored": ignored_by_type.get("current-time qa", 0),
            "past_time_ignored": ignored_by_type.get("past-time qa", 0),
            "ignore_file_used": args.ignore if args.ignore else None,
        },
        "overall": {
            "accuracy": overall_accuracy,
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

    print(f"\n✓ 统计结果已保存到: {output_file.name}")


if __name__ == "__main__":
    # ============ 在这里修改默认参数 ============
    DEFAULT_RESULT_DIR = "output_results/test/qwen3vl_k4_n1_pre0_fps1_debug2"
    DEFAULT_IGNORE_FILE = None
    # ==========================================
    main()
