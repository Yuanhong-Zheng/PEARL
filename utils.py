"""
Common utility functions.
Includes helpers for video processing, text processing, and related tasks.
"""
import os
import re
import subprocess
from typing import Dict, List, Sequence, Union


def time_to_seconds(time_str: str) -> float:
    """
    Convert a time string to seconds.
    
    Args:
        time_str: Time string in one of the following formats:
                 - "HH:MM:SS" (for example, "01:23:45")
                 - "MM:SS" (for example, "23:45")
                 - "SS" (for example, "45")
        
    Returns:
        Number of seconds as a float
        
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
        # HH:MM:SS format
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        # MM:SS format
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 1:
        # SS format
        return float(parts[0])
    else:
        raise ValueError(f"Invalid time format: {time_str}. Supported formats: HH:MM:SS, MM:SS, SS")


def seconds_to_time(seconds: float, format: str = "HH:MM:SS") -> str:
    """
    Convert seconds to a time string.
    
    Args:
        seconds: Number of seconds
        format: Output format, either "HH:MM:SS" or "MM:SS"
        
    Returns:
        Formatted time string
        
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
        raise ValueError(f"Unsupported format: {format}")


def extract_concepts(text: str) -> List[str]:
    """
    Extract concepts from text, where concepts are enclosed in {}.
    
    Args:
        text: Input text
        
    Returns:
        List of concepts, deduplicated while preserving order
        
    Examples:
        >>> extract_concepts("This is {Concept1} and {Concept2}, plus {Concept1}")
        ['Concept1', 'Concept2']
    """
    # Use a regex to extract content in {concept_name} format
    pattern = r'\{([^}]+)\}'
    concepts = re.findall(pattern, text)
    
    # Deduplicate while preserving order
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
    Use ffmpeg to extract a clip from the source video.
    
    Args:
        source_video: Source video path
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output path
        video_codec: Video codec, default "libx264"
        audio_codec: Audio codec, default "aac"
        verbose: Whether to print verbose output
        
    Returns:
        Whether extraction succeeded
        
    Examples:
        >>> extract_video_clip("input.mp4", 10.0, 20.0, "output.mp4")
        True
    """
    try:
        # Build the ffmpeg command
        # Note: placing -ss before -i enables fast input seeking
        duration = end_time - start_time
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Input seeking for fast positioning
            '-i', source_video,
            '-t', str(duration),     # Duration rather than end time
            '-c:v', video_codec,     # Video codec
            '-c:a', audio_codec,     # Audio codec
            '-y',                    # Overwrite output file
            output_path
        ]
        
        # Execute the command
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        # Check whether the output file exists
        if os.path.exists(output_path):
            if verbose:
                print(f"✓ Video clip extracted successfully: {os.path.basename(output_path)}")
                print(f"  Time range: {start_time:.2f}s - {end_time:.2f}s")
                print(f"  Output path: {output_path}")
            return True
        else:
            if verbose:
                print("✗ Failed to extract video clip: output file does not exist")
            return False
            
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ ffmpeg failed to extract the video clip: {e}")
            print(f"  stderr: {e.stderr}")
        return False
    except Exception as e:
        if verbose:
            print(f"✗ Unexpected error while extracting video clip: {e}")
        return False


def extract_video_frame(
    source_video: str,
    timestamp: Union[str, float],
    output_path: str,
    verbose: bool = False
) -> bool:
    """
    Use ffmpeg to extract a single frame from the source video.

    Args:
        source_video: Source video path
        timestamp: Timestamp as seconds or a time string
        output_path: Output image path
        verbose: Whether to print verbose output

    Returns:
        Whether extraction succeeded
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
                print(f"✓ Frame extracted successfully: {output_path}")
            return True

        if verbose:
            print("✗ Failed to extract frame: output file does not exist")
        return False
    except subprocess.CalledProcessError as e:
        if verbose:
            print(f"✗ ffmpeg failed to extract the frame: {e}")
            print(f"stderr: {e.stderr.decode()}")
        return False
    except Exception as e:
        if verbose:
            print(f"✗ Unexpected error while extracting frame: {e}")
        return False


def get_video_duration(video_path: str) -> float:
    """
    Get the duration of a video.
    
    Args:
        video_path: Video file path
        
    Returns:
        Video duration in seconds
        
    Raises:
        RuntimeError: If the duration cannot be obtained
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
        raise RuntimeError(f"Unable to determine video duration: {e}")


def remove_concept_markers(text: str) -> str:
    """
    Remove concept markers enclosed in {} from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with concept markers removed
        
    Examples:
        >>> remove_concept_markers("This is an example with {Concept1} and {Concept2}")
        "This is an example with  and "
    """
    return re.sub(r'\{[^}]+\}', '', text)


def extract_question_without_options(text: str) -> str:
    """
    Extract the question stem from a multiple-choice prompt, excluding options.
    
    Args:
        text: Full question text including options
        
    Returns:
        The question stem without the options
        
    Examples:
        >>> extract_question_without_options("Who was using this cup? A. {XiaoMing} B. {XiaoJing} C. No one")
        "Who was using this cup?"
        >>> extract_question_without_options("What color is it? A. Red B. Blue")
        "What color is it?"
    """
    # Use a regex to find the start of the option list (" A.", " B.", etc.)
    match = re.search(r'\s+[A-Z]\.', text)
    
    if match:
        # If found, keep only the text before the first option marker
        question_only = text[:match.start()].strip()
        return question_only
    else:
        # If no option marker is found, return the trimmed original text
        return text.strip()


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and line breaks.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Collapse consecutive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Trim leading and trailing whitespace
    text = text.strip()
    return text


def has_complete_option_fields(
    qa_item: Dict,
    option_labels: Sequence[str] = ("A", "B", "C", "D"),
) -> bool:
    """
    Check whether qa_item contains a complete set of option fields (optionA-optionD).
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
    Build the question text:
    - Use qa_item['question'] as the stem
    - If all option fields exist, append them as multi-line options
    - Otherwise:
      - return the stem when require_complete_options=False
      - raise an assertion when require_complete_options=True
    """
    stem = str(qa_item.get("question", "")).strip()
    assert stem, "qa_item['question'] cannot be an empty string"

    has_all = has_complete_option_fields(qa_item, option_labels)
    if require_complete_options:
        assert has_all, "qa_item is missing a complete set of optionA-optionD fields"
    if not has_all:
        return stem

    option_lines = []
    for label in option_labels:
        key = f"option{label}"
        value = qa_item.get(key)
        assert value is not None, f"qa_item['{key}'] cannot be empty"
        option_lines.append(f"{label}. {str(value).strip()}")

    return f"{stem}\n" + "\n".join(option_lines)


def build_rotated_qa_item(
    qa_item: Dict,
    target_gt: str,
    option_labels: Sequence[str] = ("A", "B", "C", "D"),
) -> Dict:
    """
    Build a rotated copy of the question:
    - Move the correct option content to target_gt
    - Use a minimal-change swap between the original correct option and the target option
    - Update gt to target_gt
    """
    target_gt = str(target_gt).strip().upper()
    assert target_gt in option_labels, f"Invalid target_gt: {target_gt}"

    original_gt = str(qa_item.get("gt", "")).strip().upper()
    assert original_gt in option_labels, f"Invalid original gt: {original_gt}"
    assert has_complete_option_fields(qa_item, option_labels), "Rotation requires a complete set of optionA-optionD"

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
    Extract the answer from a model response.
    
    Args:
        response: Full response text from the model
        
    Returns:
        Extracted answer option letter, or an empty string if not found
        
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
    # Step 1: try to extract the content inside <ans>...</ans>
    match = re.search(r'<ans>(.*?)</ans>', response, re.IGNORECASE | re.DOTALL)
    
    if match:
        # Extract the answer text and trim whitespace
        answer = match.group(1).strip()
        
        # Only keep the option letter (the first letter)
        # Supports both "A" and "A. yes"
        if answer:
            # Extract the first uppercase letter as the option
            option_match = re.match(r'^([A-Z])', answer.upper())
            if option_match:
                return option_match.group(1)
            # If no option letter is found, return the raw answer for backward compatibility
            return answer
        return ""
    
    # Step 2: if there is no <ans> tag, look for "A.", "B.", etc.
    option_match = re.search(r'\b([A-Z])\.', response)
    if option_match:
        return option_match.group(1)
    
    # Step 3: try to extract a standalone uppercase option letter
    single_letter_match = re.search(r'(?:^|\s)([A-D])(?:\s|$)', response)
    if single_letter_match:
        return single_letter_match.group(1)
    
    # Step 4: return an empty string if nothing matches
    return ""


def evaluate_qa_results(results_data: List[dict]) -> dict:
    """
    Evaluate QA results and compute accuracies plus error details.
    
    Args:
        results_data: QA results list; each item includes video_path and timestamps
                     Each question in timestamps should include id, qa_type, gt, and answer
        
    Returns:
        Evaluation result dict, containing:
        - total_accuracy: overall accuracy
        - current_time_accuracy: accuracy for current-time qa
        - past_time_accuracy: accuracy for past-time qa
        - total_count: total number of questions
        - correct_count: total number of correct answers
        - current_time_count: number of current-time qa questions
        - current_time_correct: number of correct current-time qa answers
        - past_time_count: number of past-time qa questions
        - past_time_correct: number of correct past-time qa answers
        - wrong_ids: list of wrong question ids
        - details: per-question details (id, qa_type, gt, predicted, is_correct)
        
    Examples:
        >>> results = [{"video_path": "...", "timestamps": [...]}]
        >>> stats = evaluate_qa_results(results)
        >>> print(f"Overall accuracy: {stats['total_accuracy']:.2%}")
    """
    # Initialize counters
    total_count = 0
    correct_count = 0
    current_time_count = 0
    current_time_correct = 0
    past_time_count = 0
    past_time_correct = 0
    wrong_ids = []
    details = []
    
    # Iterate over QA results for all videos
    for video_item in results_data:
        timestamps = video_item.get('timestamps', [])
        
        for qa_item in timestamps:
            qa_id = qa_item.get('id')
            qa_type = qa_item.get('qa_type')
            gt = qa_item.get('gt', '').strip()
            answer_raw = qa_item.get('answer', '')
            
            # Extract the answer from the model response
            predicted = extract_answer_from_response(answer_raw).strip()
            
            # Check correctness case-insensitively
            is_correct = (predicted.upper() == gt.upper()) if (predicted and gt) else False
            
            # Record detailed information
            details.append({
                'id': qa_id,
                'qa_type': qa_type,
                'gt': gt,
                'predicted': predicted,
                'is_correct': is_correct
            })
            
            # Update overall counts
            total_count += 1
            if is_correct:
                correct_count += 1
            else:
                wrong_ids.append(qa_id)
            
            # Update per-category counts
            if qa_type == 'current-time qa':
                current_time_count += 1
                if is_correct:
                    current_time_correct += 1
            elif qa_type == 'past-time qa':
                past_time_count += 1
                if is_correct:
                    past_time_correct += 1
    
    # Compute accuracies
    total_accuracy = correct_count / total_count if total_count > 0 else 0.0
    current_time_accuracy = current_time_correct / current_time_count if current_time_count > 0 else 0.0
    past_time_accuracy = past_time_correct / past_time_count if past_time_count > 0 else 0.0
    
    # Return the evaluation result
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
    Print a formatted evaluation report.
    
    Args:
        eval_result: Evaluation result dict returned by evaluate_qa_results
    """
    print("\n" + "=" * 80)
    print("QA Evaluation Report")
    print("=" * 80)
    
    # Overall statistics
    print("\n[Overall]")
    print(f"  Total questions: {eval_result['total_count']}")
    print(f"  Correct: {eval_result['correct_count']}")
    print(f"  Incorrect: {len(eval_result['wrong_ids'])}")
    print(f"  Overall accuracy: {eval_result['total_accuracy']:.2%}")
    
    # Current-time QA statistics
    print("\n[Current-time QA]")
    print(f"  Questions: {eval_result['current_time_count']}")
    print(f"  Correct: {eval_result['current_time_correct']}")
    print(f"  Incorrect: {eval_result['current_time_count'] - eval_result['current_time_correct']}")
    print(f"  Accuracy: {eval_result['current_time_accuracy']:.2%}")
    
    # Past-time QA statistics
    print("\n[Past-time QA]")
    print(f"  Questions: {eval_result['past_time_count']}")
    print(f"  Correct: {eval_result['past_time_correct']}")
    print(f"  Incorrect: {eval_result['past_time_count'] - eval_result['past_time_correct']}")
    print(f"  Accuracy: {eval_result['past_time_accuracy']:.2%}")
    
    # Incorrect question list
    if eval_result['wrong_ids']:
        print("\n[Incorrect Question IDs]")
        print(f"  {eval_result['wrong_ids']}")
        
        # Show details for incorrect questions
        print("\n[Incorrect Question Details]")
        for detail in eval_result['details']:
            if not detail['is_correct']:
                print(f"  ID {detail['id']} ({detail['qa_type']}): GT={detail['gt']}, Predicted={detail['predicted']}")
    else:
        print("\n[Incorrect Question IDs]")
        print("  No incorrect questions. Everything is correct!")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Simple tests
    print("Testing time conversion:")
    print(f"  time_to_seconds('01:23:45') = {time_to_seconds('01:23:45')}")
    print(f"  time_to_seconds('23:45') = {time_to_seconds('23:45')}")
    print(f"  seconds_to_time(5025) = {seconds_to_time(5025)}")
    
    print("\nTesting concept extraction:")
    test_text = "This question is about {ConceptA} and {ConceptB}, and {ConceptA} is important"
    concepts = extract_concepts(test_text)
    print(f"  Extracted concepts: {concepts}")
    
    print("\nTesting text cleaning:")
    test_text = "This   is   a question   about   {ConceptA}"
    print(f"  Original text: '{test_text}'")
    print(f"  Without markers: '{remove_concept_markers(test_text)}'")
    print(f"  Cleaned text: '{clean_text(remove_concept_markers(test_text))}'")
    
    print("\nTesting question extraction:")
    test_q1 = "Who was using this cup just now? A. {XiaoMing} B. {XiaoJing} C. No one"
    test_q2 = "What color is the car? A. Red B. Blue C. Green"
    print(f"  Original question 1: '{test_q1}'")
    print(f"  Extracted stem: '{extract_question_without_options(test_q1)}'")
    print(f"  Original question 2: '{test_q2}'")
    print(f"  Extracted stem: '{extract_question_without_options(test_q2)}'")
