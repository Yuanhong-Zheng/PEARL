"""
通用工具函数模块
包含视频处理、文本处理等常用功能
"""
import os
import re
import subprocess
from typing import Dict, List, Sequence, Union


def time_to_seconds(time_str: str) -> float:
    """
    将时间字符串转换为秒数
    
    Args:
        time_str: 时间字符串，支持以下格式：
                 - "HH:MM:SS" (如 "01:23:45")
                 - "MM:SS" (如 "23:45")
                 - "SS" (如 "45")
        
    Returns:
        秒数（float）
        
    Examples:
        >>> time_to_seconds("01:23:45")
        5025.0
        >>> time_to_seconds("23:45")
        1425.0
        >>> time_to_seconds("45")
        45.0
    """
    parts = time_str.split(':')
    
    if len(parts) == 3:
        # HH:MM:SS 格式
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        # MM:SS 格式
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 1:
        # SS 格式
        return float(parts[0])
    else:
        raise ValueError(f"无效的时间格式: {time_str}，支持的格式: HH:MM:SS, MM:SS, SS")


def seconds_to_time(seconds: float, format: str = "HH:MM:SS") -> str:
    """
    将秒数转换为时间字符串
    
    Args:
        seconds: 秒数
        format: 输出格式，可选 "HH:MM:SS" 或 "MM:SS"
        
    Returns:
        时间字符串
        
    Examples:
        >>> seconds_to_time(5025.5)
        "01:23:45"
        >>> seconds_to_time(1425, format="MM:SS")
        "23:45"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if format == "HH:MM:SS":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    elif format == "MM:SS":
        total_minutes = int(seconds // 60)
        return f"{total_minutes:02d}:{secs:02d}"
    else:
        raise ValueError(f"不支持的格式: {format}")


def extract_concepts(text: str) -> List[str]:
    """
    从文本中提取概念（用 {} 包围的内容）
    
    Args:
        text: 输入文本
        
    Returns:
        概念列表（去重且保持顺序）
        
    Examples:
        >>> extract_concepts("这是 {概念1} 和 {概念2}，还有 {概念1}")
        ['概念1', '概念2']
    """
    # 使用正则表达式提取 {concept_name} 格式的内容
    pattern = r'\{([^}]+)\}'
    concepts = re.findall(pattern, text)
    
    # 去重，同时保持原有顺序
    seen = set()
    unique_concepts = []
    for concept in concepts:
        if concept not in seen:
            seen.add(concept)
            unique_concepts.append(concept)
    
    return unique_concepts


def extract_video_clip(
    source_video: str,
    start_time: float,
    end_time: float,
    output_path: str,
    video_codec: str = "libx264",
    audio_codec: str = "aac",
    verbose: bool = False
) -> bool:
    """
    使用 ffmpeg 从源视频中提取指定时间段的片段
    
    Args:
        source_video: 源视频路径
        start_time: 开始时间（秒）
        end_time: 结束时间（秒）
        output_path: 输出路径
        video_codec: 视频编码器，默认 "libx264"
        audio_codec: 音频编码器，默认 "aac"
        verbose: 是否显示详细输出
        
    Returns:
        是否成功
        
    Examples:
        >>> extract_video_clip("input.mp4", 10.0, 20.0, "output.mp4")
        True
    """
    try:
        # 构造 ffmpeg 命令
        # 注意：-ss 放在 -i 之前可以实现快速 seek（Input Seeking）
        # 这样无论 start_time 在视频的哪个位置，速度都很快
        duration = end_time - start_time
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Input Seeking：快速定位
            '-i', source_video,
            '-t', str(duration),     # 持续时间（而不是结束时间）
            '-c:v', video_codec,     # 视频编码
            '-c:a', audio_codec,     # 音频编码
            '-y',                    # 覆盖输出文件
            output_path
        ]
        
        # 执行命令
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # 检查输出文件是否存在
        if os.path.exists(output_path):
            if verbose:
                print(f"✓ 成功提取视频片段: {os.path.basename(output_path)}")
                print(f"  时间范围: {start_time:.2f}s - {end_time:.2f}s")
                print(f"  输出路径: {output_path}")
            return True
        else:
            if verbose:
                print(f"✗ 提取视频片段失败: 输出文件不存在")
            return False
            
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ ffmpeg 提取视频片段失败: {e}")
            print(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        if verbose:
            print(f"✗ 提取视频片段异常: {e}")
        return False


def extract_video_frame(
    source_video: str,
    timestamp: Union[str, float],
    output_path: str,
    verbose: bool = False
) -> bool:
    """
    使用 ffmpeg 从源视频中提取指定时间点的一帧图像。

    Args:
        source_video: 源视频路径
        timestamp: 时间点（秒或时间字符串）
        output_path: 输出图像路径
        verbose: 是否显示详细输出

    Returns:
        是否成功
    """
    try:
        cmd = [
            "ffmpeg",
            "-i", source_video,
            "-ss", str(timestamp),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            output_path,
        ]

        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
        )

        if os.path.exists(output_path):
            if verbose:
                print(f"✓ 成功提取帧: {output_path}")
            return True

        if verbose:
            print("✗ 提取帧失败: 输出文件不存在")
        return False
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ ffmpeg 提取帧失败: {e}")
            print(f"stderr: {e.stderr.decode()}")
        return False
    except Exception as e:
        if verbose:
            print(f"✗ 提取帧异常: {e}")
        return False


def get_video_duration(video_path: str) -> float:
    """
    获取视频时长
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        视频时长（秒）
        
    Raises:
        RuntimeError: 如果无法获取视频时长
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        duration = float(result.stdout.strip())
        return duration
        
    except Exception as e:
        raise RuntimeError(f"无法获取视频时长: {e}")


def remove_concept_markers(text: str) -> str:
    """
    从文本中移除概念标记（{} 包围的部分）
    
    Args:
        text: 输入文本
        
    Returns:
        移除概念标记后的文本
        
    Examples:
        >>> remove_concept_markers("这是 {概念1} 和 {概念2} 的例子")
        "这是  和  的例子"
    """
    return re.sub(r'\{[^}]+\}', '', text)


def extract_question_without_options(text: str) -> str:
    """
    从选择题中提取问题部分，不包含选项
    
    Args:
        text: 完整的问题文本（包含选项）
        
    Returns:
        只包含问题的文本（不含选项）
        
    Examples:
        >>> extract_question_without_options("Who was using this cup? A. {XiaoMing} B. {XiaoJing} C. No one")
        "Who was using this cup?"
        >>> extract_question_without_options("What color is it? A. Red B. Blue")
        "What color is it?"
    """
    # 使用正则表达式匹配选项的开始位置（空格 + 大写字母 + 点，如 " A."）
    # 匹配模式：可能有多个空格，然后是大写字母A-Z，然后是点
    match = re.search(r'\s+[A-Z]\.', text)
    
    if match:
        # 找到选项开始位置，截取之前的内容
        question_only = text[:match.start()].strip()
        return question_only
    else:
        # 没有找到选项标记，返回原文本（去除首尾空格）
        return text.strip()


def clean_text(text: str) -> str:
    """
    清理文本（移除多余空格、换行等）
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 移除首尾空格
    text = text.strip()
    return text


def has_complete_option_fields(
    qa_item: Dict,
    option_labels: Sequence[str] = ("A", "B", "C", "D"),
) -> bool:
    """
    判断 qa_item 是否包含完整的 option 字段（optionA~optionD）。
    """
    for label in option_labels:
        key = f"option{label}"
        if key not in qa_item or qa_item.get(key) is None:
            return False
    return True


def build_question_with_options(
    qa_item: Dict,
    option_labels: Sequence[str] = ("A", "B", "C", "D"),
    require_complete_options: bool = False,
) -> str:
    """
    构造问题文本：
    - 基于 qa_item['question'] 作为 stem；
    - 若有完整 option 字段，则拼接成多行选项；
    - 若无完整 option 字段：
      - require_complete_options=False 时返回 stem（兼容旧数据）
      - require_complete_options=True 时抛出断言
    """
    stem = str(qa_item.get("question", "")).strip()
    assert stem, "qa_item['question'] 不能为空字符串"

    has_all = has_complete_option_fields(qa_item, option_labels)
    if require_complete_options:
        assert has_all, "qa_item 缺少完整 optionA~optionD 字段"
    if not has_all:
        return stem

    option_lines = []
    for label in option_labels:
        key = f"option{label}"
        value = qa_item.get(key)
        assert value is not None, f"qa_item['{key}'] 不能为空"
        option_lines.append(f"{label}. {str(value).strip()}")

    return f"{stem}\n" + "\n".join(option_lines)


def build_rotated_qa_item(
    qa_item: Dict,
    target_gt: str,
    option_labels: Sequence[str] = ("A", "B", "C", "D"),
) -> Dict:
    """
    构造轮换后的题目副本：
    - 正确内容移动到 target_gt 对应位置；
    - 采用“交换原正确选项与目标选项”的最小扰动策略；
    - 更新 gt 为 target_gt。
    """
    target_gt = str(target_gt).strip().upper()
    assert target_gt in option_labels, f"无效 target_gt: {target_gt}"

    original_gt = str(qa_item.get("gt", "")).strip().upper()
    assert original_gt in option_labels, f"无效原始 gt: {original_gt}"
    assert has_complete_option_fields(qa_item, option_labels), "轮换需要完整 optionA~optionD"

    rotated = qa_item.copy()
    option_map = {label: str(qa_item[f"option{label}"]) for label in option_labels}

    if target_gt != original_gt:
        correct_content = option_map[original_gt]
        target_content = option_map[target_gt]
        option_map[target_gt] = correct_content
        option_map[original_gt] = target_content

    for label in option_labels:
        rotated[f"option{label}"] = option_map[label]
    rotated["gt"] = target_gt
    return rotated


def extract_answer_from_response(response: str) -> str:
    """
    从模型回答中提取答案
    
    Args:
        response: 模型的完整回答文本
        
    Returns:
        提取的答案选项字母（如 "A", "B", "C" 等），如果未找到则返回空字符串
        
    Examples:
        >>> extract_answer_from_response("I think the answer is <ans>A</ans>")
        "A"
        >>> extract_answer_from_response("The correct choice is <ans>B. yes</ans> because...")
        "B"
        >>> extract_answer_from_response("B. his shirt")
        "B"
        >>> extract_answer_from_response("A")
        "A"
        >>> extract_answer_from_response("I don't know")
        ""
    """
    # 第一步：尝试提取 <ans>...</ans> 标签内的内容
    match = re.search(r'<ans>(.*?)</ans>', response, re.IGNORECASE | re.DOTALL)
    
    if match:
        # 提取答案并清理空格
        answer = match.group(1).strip()
        
        # 只提取选项字母部分（第一个字母）
        # 兼容 "A" 和 "A. yes" 两种格式
        if answer:
            # 提取第一个大写字母作为选项
            option_match = re.match(r'^([A-Z])', answer.upper())
            if option_match:
                return option_match.group(1)
            # 如果没有找到字母，返回原始答案（向后兼容）
            return answer
        return ""
    
    # 第二步：如果没有 <ans> 标签，尝试提取 "A.", "B.", "C.", "D." 格式的答案
    # 匹配以大写字母开头，后面紧跟一个点的模式
    option_match = re.search(r'\b([A-Z])\.', response)
    if option_match:
        return option_match.group(1)
    
    # 第三步：尝试提取单个大写字母（如 "A", "B", "C", "D"）
    # 匹配独立的大写字母（前后是单词边界或空格）
    single_letter_match = re.search(r'(?:^|\s)([A-D])(?:\s|$)', response)
    if single_letter_match:
        return single_letter_match.group(1)
    
    # 第四步：如果都没找到，返回空字符串
    return ""


def evaluate_qa_results(results_data: List[dict]) -> dict:
    """
    评估问答结果，统计正确率和错误题目
    
    Args:
        results_data: 问答结果列表，每个元素是包含 video_path 和 timestamps 的字典
                     timestamps 中每个问题应包含：id, qa_type, gt, answer 字段
        
    Returns:
        评估结果字典，包含：
        - total_accuracy: 总正确率
        - current_time_accuracy: current-time qa 正确率
        - past_time_accuracy: past-time qa 正确率
        - total_count: 总题目数
        - correct_count: 总正确数
        - current_time_count: current-time qa 题目数
        - current_time_correct: current-time qa 正确数
        - past_time_count: past-time qa 题目数
        - past_time_correct: past-time qa 正确数
        - wrong_ids: 错误题目的 id 列表
        - details: 每道题的详细信息（id, qa_type, gt, predicted, is_correct）
        
    Examples:
        >>> results = [{"video_path": "...", "timestamps": [...]}]
        >>> stats = evaluate_qa_results(results)
        >>> print(f"总正确率: {stats['total_accuracy']:.2%}")
    """
    # 初始化统计变量
    total_count = 0
    correct_count = 0
    current_time_count = 0
    current_time_correct = 0
    past_time_count = 0
    past_time_correct = 0
    wrong_ids = []
    details = []
    
    # 遍历所有视频的问答结果
    for video_item in results_data:
        timestamps = video_item.get('timestamps', [])
        
        for qa_item in timestamps:
            qa_id = qa_item.get('id')
            qa_type = qa_item.get('qa_type')
            gt = qa_item.get('gt', '').strip()
            answer_raw = qa_item.get('answer', '')
            
            # 从模型回答中提取答案
            predicted = extract_answer_from_response(answer_raw).strip()
            
            # 判断是否正确（不区分大小写）
            is_correct = (predicted.upper() == gt.upper()) if (predicted and gt) else False
            
            # 记录详细信息
            details.append({
                'id': qa_id,
                'qa_type': qa_type,
                'gt': gt,
                'predicted': predicted,
                'is_correct': is_correct
            })
            
            # 统计总数
            total_count += 1
            if is_correct:
                correct_count += 1
            else:
                wrong_ids.append(qa_id)
            
            # 分类统计
            if qa_type == 'current-time qa':
                current_time_count += 1
                if is_correct:
                    current_time_correct += 1
            elif qa_type == 'past-time qa':
                past_time_count += 1
                if is_correct:
                    past_time_correct += 1
    
    # 计算正确率
    total_accuracy = correct_count / total_count if total_count > 0 else 0.0
    current_time_accuracy = current_time_correct / current_time_count if current_time_count > 0 else 0.0
    past_time_accuracy = past_time_correct / past_time_count if past_time_count > 0 else 0.0
    
    # 返回结果
    return {
        'total_accuracy': total_accuracy,
        'current_time_accuracy': current_time_accuracy,
        'past_time_accuracy': past_time_accuracy,
        'total_count': total_count,
        'correct_count': correct_count,
        'current_time_count': current_time_count,
        'current_time_correct': current_time_correct,
        'past_time_count': past_time_count,
        'past_time_correct': past_time_correct,
        'wrong_ids': wrong_ids,
        'details': details
    }


def print_evaluation_report(eval_result: dict):
    """
    打印评估报告（格式化输出）
    
    Args:
        eval_result: evaluate_qa_results 函数返回的评估结果字典
    """
    print("\n" + "=" * 80)
    print("问答评估报告")
    print("=" * 80)
    
    # 总体统计
    print(f"\n【总体统计】")
    print(f"  总题目数: {eval_result['total_count']}")
    print(f"  正确数: {eval_result['correct_count']}")
    print(f"  错误数: {len(eval_result['wrong_ids'])}")
    print(f"  总正确率: {eval_result['total_accuracy']:.2%}")
    
    # Current-time QA 统计
    print(f"\n【Current-time QA】")
    print(f"  题目数: {eval_result['current_time_count']}")
    print(f"  正确数: {eval_result['current_time_correct']}")
    print(f"  错误数: {eval_result['current_time_count'] - eval_result['current_time_correct']}")
    print(f"  正确率: {eval_result['current_time_accuracy']:.2%}")
    
    # Past-time QA 统计
    print(f"\n【Past-time QA】")
    print(f"  题目数: {eval_result['past_time_count']}")
    print(f"  正确数: {eval_result['past_time_correct']}")
    print(f"  错误数: {eval_result['past_time_count'] - eval_result['past_time_correct']}")
    print(f"  正确率: {eval_result['past_time_accuracy']:.2%}")
    
    # 错误题目列表
    if eval_result['wrong_ids']:
        print(f"\n【错误题目 ID】")
        print(f"  {eval_result['wrong_ids']}")
        
        # 显示错误题目的详细信息
        print(f"\n【错误题目详情】")
        for detail in eval_result['details']:
            if not detail['is_correct']:
                print(f"  ID {detail['id']} ({detail['qa_type']}): GT={detail['gt']}, Predicted={detail['predicted']}")
    else:
        print(f"\n【错误题目 ID】")
        print(f"  无错误题目，全部正确！")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 简单测试
    print("测试时间转换功能:")
    print(f"  time_to_seconds('01:23:45') = {time_to_seconds('01:23:45')}")
    print(f"  time_to_seconds('23:45') = {time_to_seconds('23:45')}")
    print(f"  seconds_to_time(5025) = {seconds_to_time(5025)}")
    
    print("\n测试概念提取功能:")
    test_text = "这是关于 {概念A} 和 {概念B} 的问题，{概念A} 很重要"
    concepts = extract_concepts(test_text)
    print(f"  提取的概念: {concepts}")
    
    print("\n测试文本清理功能:")
    test_text = "这是   关于   {概念A}   的问题"
    print(f"  原文本: '{test_text}'")
    print(f"  移除标记: '{remove_concept_markers(test_text)}'")
    print(f"  清理后: '{clean_text(remove_concept_markers(test_text))}'")
    
    print("\n测试问题提取功能:")
    test_q1 = "Who was using this cup just now? A. {XiaoMing} B. {XiaoJing} C. No one"
    test_q2 = "What color is the car? A. Red B. Blue C. Green"
    print(f"  原问题1: '{test_q1}'")
    print(f"  提取后: '{extract_question_without_options(test_q1)}'")
    print(f"  原问题2: '{test_q2}'")
    print(f"  提取后: '{extract_question_without_options(test_q2)}'")
