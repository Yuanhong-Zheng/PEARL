"""
Video QA inference script.
Integrates ConceptDatabase and ClipMemory for Qwen3-VL-based question answering.
"""
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI

from concept_database import ConceptDatabase
from clip_memory import ClipMemory
from utils import (
    time_to_seconds, extract_concepts, extract_video_clip, 
    extract_question_without_options,
    evaluate_qa_results, print_evaluation_report,
    build_question_with_options, build_rotated_qa_item,
    has_complete_option_fields, extract_answer_from_response
)
from concept_desc import generate_distinctive_description


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a path to an absolute path; relative paths are based on base_dir."""
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _to_file_url(path_str: str) -> str:
    """Convert a local path to a standard file:// URL."""
    return Path(path_str).resolve().as_uri()


class VideoQAInference:
    """Video QA inference system."""
    
    def __init__(
        self,
        annotation_path: str,
        clips_info_path: str,
        cache_dir: str = "./.cache",
        api_base_url: str = "http://127.0.0.1:22003/v1",
        model_path: str = "models/Qwen3-VL-8B-Instruct",
        use_video_embedding: bool = False,
        embedding_api_url: str = "http://localhost:5000",
        batch_size: int = 10,
        output_dir: str = "./.cache/qa_output",
        clear_concept_db: bool = True,
        num_neighbor: int = 1,
        enable_rotation: bool = True
    ):
        """
        Initialize the inference system.
        
        Args:
            annotation_path: Annotation file path used to extract concept definitions
            clips_info_path: Path to the clips metadata JSON file
            cache_dir: Cache directory for the concept database and extracted frames
            api_base_url: API service URL
            model_path: Model path
            use_video_embedding: Whether to retrieve with video embeddings instead of text descriptions
            embedding_api_url: Embedding API service URL
            batch_size: Batch size for embedding precomputation
            output_dir: Output directory for temporary video clips
            clear_concept_db: Whether to clear the concept database before insertion
            num_neighbor: Number of neighboring clips to include
            enable_rotation: Whether to enable option-rotation evaluation
        """
        print("=" * 80)
        print("Initializing the video QA inference system...")
        print("=" * 80)

        self.annotation_path = str(Path(annotation_path).resolve())
        self.clips_info_path = str(Path(clips_info_path).resolve())
        self.annotation_base_dir = Path(self.annotation_path).parent
        model_path = str(_resolve_path(model_path, Path(__file__).resolve().parent))
        
        self.num_neighbor = num_neighbor
        self.enable_rotation = enable_rotation
        
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print("\n[1/3] Initializing concept database...")

        annotation_cache_dir = Path(cache_dir).resolve() / Path(annotation_path).stem
        annotation_cache_dir.mkdir(parents=True, exist_ok=True)

        self.concept_db = ConceptDatabase(
            db_path=str(annotation_cache_dir / "concept_db.json"),
            frame_dir=str(annotation_cache_dir),
        )

        self.concept_db.add_concepts_from_annotation_file(
            annotation_file=self.annotation_path,
            clear_before_add=clear_concept_db
        )
        print(f"✓ Concept database initialized with {len(self.concept_db.data['concepts'])} concept(s)")
        
        temp_client = OpenAI(api_key="EMPTY", base_url=api_base_url, timeout=3600)
        
        for concept in self.concept_db.data['concepts']:
            concept['retrieval_description'] = generate_distinctive_description(
                client=temp_client,
                model_path=model_path,
                image_path=concept.get('frame_path', ''),
                concept_name=concept.get('concept_name', 'Unknown'),
                original_description=concept.get('description', ''),
            )

        self.concept_db._save_db()
        self.concept_retrieval_map = {
            concept.get('concept_name', ''): concept.get('retrieval_description', '')
            for concept in self.concept_db.data['concepts']
            if concept.get('concept_name', '') and concept.get('retrieval_description', '')
        }

        print("\n[2/3] Initializing clip memory system...")
        embeddings_cache_dir = annotation_cache_dir
        
        self.clip_memory = ClipMemory(
            json_path=self.clips_info_path,
            api_base_url=embedding_api_url,
            use_video_embedding=use_video_embedding,
            batch_size=batch_size,
            cache_dir=str(embeddings_cache_dir),
            force_recompute=False
        )
        print(f"✓ Clip memory system loaded with {len(self.clip_memory.clips_data)} clip(s)")

        print("\n[3/3] Initializing model API...")
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_base_url,
            timeout=3600
        )
        self.model_path = model_path
        print(f"✓ Model API initialized: {self.model_path}")
        
        print("\n" + "=" * 80)
        print("Initialization complete!")
        print("=" * 80 + "\n")
    
    
    def retrieve_concept_info(self, concept_name: str) -> Dict:
        """
        Retrieve concept information from the concept database.
        
        Args:
            concept_name: Concept name
            
        Returns:
            Concept information dict containing frame_path, concept_name, etc.
        """
        if (concept_info := self.concept_db.query_by_name(concept_name)) is None:
            print(f"  ⚠ Warning: concept '{concept_name}' was not found in the database")
            import ipdb;ipdb.set_trace()
            return None

        if (frame_path := concept_info.get("frame_path")) and not Path(frame_path).is_absolute():
            concept_info = concept_info.copy()
            concept_info["frame_path"] = str((Path(self.concept_db.db_path).parent / frame_path).resolve())
        return concept_info
    
    
    def get_clip_at_time(self, time_seconds: float) -> Optional[Dict]:
        """
        Get the video clip containing a specified timestamp.
        
        Args:
            time_seconds: Timestamp in seconds
            
        Returns:
            Clip info dict, or None if no clip contains the timestamp
        """
        for clip in self.clip_memory.clips_data:
            if clip['start_time'] <= time_seconds < clip['end_time']:
                return clip.copy()
        return None
    
    
    def replace_concepts_with_descriptions(self, query: str) -> str:
        """
        Use the model to replace {ConceptName} in the query with the matching
        retrieval_description while keeping the sentence natural.
        
        Args:
            query: Original query containing {ConceptName}
            
        Returns:
            Rewritten query with natural language replacements
        """
        concepts_in_query = extract_concepts(query)
        if not concepts_in_query:
            return query
        
        replacements = {
            name: self.concept_retrieval_map[name]
            for name in concepts_in_query
            if name in self.concept_retrieval_map
        }
        
        if not replacements:
            print("  ⚠ No retrieval_description found for any concept; using the original query")
            return query
        
        replacement_instructions = "\n".join(
            [f'  - "{{{name}}}" should be replaced with "{desc}"' for name, desc in replacements.items()]
        )
        
        prompt = f"""Rewrite the following question by replacing the concept names (in curly braces) with their visual descriptions. Keep the sentence grammatically correct and natural.

Original question:
{query}

Replacement rules:
{replacement_instructions}

Requirements:
- Replace each {{ConceptName}} with the provided visual description
- Adjust grammar as needed (e.g., articles, verb forms) to keep the sentence natural
- Do NOT change the meaning of the question
- Do NOT add or remove any information
- Output ONLY the rewritten question, nothing else"""
        
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            max_tokens=512,
            temperature=0
        )
        rewritten_query = response.choices[0].message.content.strip()
        return rewritten_query

    
    def retrieve_relevant_clips(
        self,
        question: str,
        max_time: str = None,
        top_k: int = 1
    ):
        """
        Retrieve relevant video clips for a question.
        
        Args:
            question: Question text
            max_time: Maximum time limit in HH:MM:SS format; only clips with end_time < max_time are searched
            top_k: Number of top relevant clips to return
            
        Returns:
            List of clip information dicts
        """
        query = extract_question_without_options(question).strip()
        if self.concept_retrieval_map:
            query = self.replace_concepts_with_descriptions(query)
        if max_time:
            max_seconds = time_to_seconds(max_time)
            original_clips_data = self.clip_memory.clips_data
            original_clip_embeddings = self.clip_memory.clip_embeddings
            filtered = [(i, clip) for i, clip in enumerate(original_clips_data) if clip['end_time'] < max_seconds]
            
            if not filtered:
                print(f"  ⚠ Warning: no clips were found before time {max_time}")
                return []

            filtered_indices, filtered_clips = zip(*filtered)
            self.clip_memory.clips_data = filtered_clips
            self.clip_memory.clip_embeddings = original_clip_embeddings[list(filtered_indices)]
            results = self.clip_memory.search(query, top_k=top_k)

            self.clip_memory.clips_data = original_clips_data
            self.clip_memory.clip_embeddings = original_clip_embeddings
        else:
            results = self.clip_memory.search(query, top_k=top_k)
        return results
    
    def expand_clips_with_neighbors(self, retrieved_clips: List[Dict], current_clip_start_time: float = None) -> List[Dict]:
        """
        Expand retrieved clips by including neighboring clips before and after each result.
        
        Args:
            retrieved_clips: Retrieved clip list
            current_clip_start_time: Start time of the current clip, used to avoid overlap
            
        Returns:
            Expanded clip list, deduplicated and sorted by time
        """
        if not retrieved_clips:
            return []
        
        if self.num_neighbor == 0:
            expanded_clips = sorted((clip.copy() for clip in retrieved_clips), key=lambda x: x['start_time'])
            if current_clip_start_time is None:
                return expanded_clips
            original_count = len(expanded_clips)
            expanded_clips = [clip for clip in expanded_clips if clip['end_time'] <= current_clip_start_time]
            if (filtered_count := original_count - len(expanded_clips)) > 0:
                print(f"  Filtered out {filtered_count} clip(s) overlapping with the current clip")
            return expanded_clips

        clip_id_to_index = {clip['clip_id']: i for i, clip in enumerate(self.clip_memory.clips_data)}
        expanded_clip_ids = set()
        
        for clip in retrieved_clips:
            clip_id = clip['clip_id']
            expanded_clip_ids.add(clip_id)

            if clip_id not in clip_id_to_index:
                print(f"      ⚠ Warning: Clip ID {clip_id} was not found in the data")
                continue
            
            current_index = clip_id_to_index[clip_id]
            expanded_clip_ids.update(
                self.clip_memory.clips_data[i]['clip_id']
                for i in range(max(0, current_index - self.num_neighbor), min(len(self.clip_memory.clips_data), current_index + self.num_neighbor + 1))
            )

        expanded_clips = sorted(
            (clip.copy() for clip in self.clip_memory.clips_data if clip['clip_id'] in expanded_clip_ids),
            key=lambda x: x['start_time']
        )
        if current_clip_start_time is not None:
            original_count = len(expanded_clips)
            expanded_clips = [clip for clip in expanded_clips if clip['end_time'] <= current_clip_start_time]
            if (filtered_count := original_count - len(expanded_clips)) > 0:
                print(f"  Filtered out {filtered_count} clip(s) overlapping with the current clip")
        
        return expanded_clips
    
    def build_messages(
        self,
        question: str,
        concepts_info: List[Dict],
        clips_info: List[Dict],
        current_clip_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Build the messages sent to the model.
        
        Args:
            question: Question text
            concepts_info: Concept info list
            clips_info: Historical related clips
            current_clip_path: Current video clip path, from the clip start to the question timestamp
            
        Returns:
            Message list
        """
        content, add_media = [], lambda media_type, path: content.append({
            "type": media_type,
            media_type: {"url": _to_file_url(path)},
        })
        for concept in concepts_info:
            if concept is None:
                continue
            add_media("video_url" if concept.get('concept_type', 'frame') == 'clip' else "image_url", concept["frame_path"])
            content.append({"type": "text", "text": concept['description']})
        
        if clips_info:
            content.append({"type": "text", "text": "Here are the previous related video clips:"})
            for clip in clips_info:
                add_media("video_url", clip["clip_path"])
        
        if current_clip_path:
            content.append({"type": "text", "text": "The following is the current video clip:"})
            add_media("video_url", current_clip_path)

        content.append({"type": "text", "text": f"{question}\n\nPlease output your answer choice in <ans></ans> tags."})
        return [{"role": "user", "content": content}]
    
    def call_model(self, messages: List[Dict]) -> tuple:
        """
        Call the model for inference.
        
        Args:
            messages: Message list
            
        Returns:
            Model-generated answer
        """
        return self.client.chat.completions.create(
            model=self.model_path, messages=messages, max_tokens=1024, temperature=0
        ).choices[0].message.content
    
    def process_qa(self, qa_item: Dict, top_k_clips: int = 1) -> Dict:
        """
        Process a single QA item.
        
        Args:
            qa_item: QA item dictionary
            top_k_clips: Number of clips to retrieve
            
        Returns:
            QA item with an added answer field
        """
        question = build_question_with_options(qa_item, require_complete_options=False)
        qa_id = qa_item['id']
        qa_type = qa_item['qa_type']
        
        print(f"\n{'='*80}")
        print(f"Processing question ID: {qa_id}")
        print(f"Question type: {qa_type}")
        print(f"Time: {qa_item.get('time') or qa_item.get('end_time')}")
        print(f"Question: {question}")
        print(f"{'='*80}")

        print("\n[Step 1] Extracting concepts...")
        concepts = extract_concepts(question)

        print("\n[Step 2] Retrieving concept information...")
        assert concepts, "No concepts were extracted from the question. Please verify the format; concepts should be enclosed in {}"
        concepts_info = [info for name in concepts if (info := self.retrieve_concept_info(name))]
        question_time = qa_item.get('time') or qa_item.get('end_time')
        time_seconds = time_to_seconds(question_time)

        print(f"\n[Step 3] Retrieving relevant video clips (Top {top_k_clips})...")
        clips_info = self.expand_clips_with_neighbors(
            self.retrieve_relevant_clips(question, max_time=question_time, top_k=top_k_clips),
            (current_clip['start_time'] if (current_clip := self.get_clip_at_time(time_seconds)) else max(0, time_seconds - 1)),
        )

        print("\n[Step 4] Extracting the current video clip...")
        if current_clip is None:
            print(f"  ⚠ No matching video clip found at {question_time}; using the preceding 1 second as a fallback")
            clip_start_time = max(0, time_seconds - 1)
            clip_id_str = "fallback"
            print(f"  Fallback clip range: {clip_start_time:.2f}s - {time_seconds:.2f}s")
        else:
            clip_start_time = current_clip['start_time']
            clip_id_str = str(current_clip['clip_id'])

        current_clip_path = str((self.output_dir / f"qa_{qa_id}_current_{clip_id_str}_{clip_start_time:.2f}_{time_seconds:.2f}.mp4").resolve())
        success = extract_video_clip(
            source_video=self.clip_memory.source_video,
            start_time=clip_start_time,
            end_time=time_seconds,
            output_path=current_clip_path,
            verbose=True
        )
        assert success

        print("\n[Step 5] Building model input...")
        messages = self.build_messages(question, concepts_info, clips_info, current_clip_path)
        rotation_enabled_for_qa = (
            self.enable_rotation
            and has_complete_option_fields(qa_item)
            and str(qa_item.get("gt", "")).strip().upper() in ("A", "B", "C", "D")
        )
        original_gt = str(qa_item.get("gt", "")).strip().upper()
        if rotation_enabled_for_qa:
            print("\n[Step 6] Rotation evaluation enabled; running 4 inference passes...")
            rotation_details = []
            all_correct = True
            for target_gt in ("A", "B", "C", "D"):
                rotated_qa = build_rotated_qa_item(qa_item, target_gt)
                rotated_question = build_question_with_options(
                    rotated_qa, require_complete_options=True
                )
                rotated_messages = self.build_messages(
                    rotated_question, concepts_info, clips_info, current_clip_path
                )
                answer = self.call_model(rotated_messages)
                predicted = extract_answer_from_response(answer).strip().upper()
                is_correct = predicted == target_gt
                all_correct = all_correct and is_correct
                rotation_details.append({
                    "target_gt": target_gt,
                    "rotated_question": rotated_question,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "raw_answer": answer,
                })

            print(f"  All four passes correct: {all_correct}")
            final_answer = f"<ans>{original_gt}</ans>" if all_correct else ""
        else:
            print("\n[Step 6] Calling the model for inference...")
            answer = self.call_model(messages)
            print(f"  ✓ Model response: {answer}")
            final_answer = answer
            all_correct = None
            rotation_details = []

        return {
            **qa_item,
            "answer": final_answer,
            "rotation_enabled": rotation_enabled_for_qa,
            "rotation_all_correct": all_correct,
            "rotation_details": rotation_details,
            "rotation_correct_flags": [item["is_correct"] for item in rotation_details],
        }
    
    def process_annotation_file(
        self,
        annotation_path: str,
        output_path: str,
        top_k_clips: int = 1
    ):
        """
        Process the entire annotation file.
        
        Args:
            annotation_path: Input annotation file path
            output_path: Output results directory path
            top_k_clips: Number of clips to retrieve
        """
        annotation_filename = Path(annotation_path).stem
        result_filename = f"{annotation_filename}_result.json"
        output_file_path = Path(output_path) / result_filename
        
        print(f"\nStarting annotation processing: {annotation_path}")
        print(f"Output file: {output_file_path}\n")

        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        results = []
        for video_item in annotations:
            video_path = video_item['video_path']
            timestamps = video_item['timestamps']
            target_qas = [qa for qa in timestamps if qa.get('qa_type') in ['past-time qa', 'current-time qa']]
            
            past_time_count = len([qa for qa in target_qas if qa.get('qa_type') == 'past-time qa'])
            current_time_count = len([qa for qa in target_qas if qa.get('qa_type') == 'current-time qa'])
            
            print(f"\nProcessing video: {video_path}")
            print(f"Number of past-time qa questions: {past_time_count}")
            print(f"Number of current-time qa questions: {current_time_count}")
            
            results.append({
                "video_path": video_path,
                "timestamps": [self.process_qa(qa_item, top_k_clips=top_k_clips) for qa_item in target_qas],
            })

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Processing complete! Results saved to: {output_file_path}")
        print(f"{'='*80}\n")

        eval_result = evaluate_qa_results(results)
        print_evaluation_report(eval_result)

        eval_filename = f"{annotation_filename}_evaluation.json"
        eval_file_path = Path(output_path) / eval_filename
        with open(eval_file_path, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Evaluation results saved to: {eval_file_path}\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Video QA inference script")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="data/annotations/thief.json",
        help="Annotation file path"
    )
    parser.add_argument(
        "--clips_base_dir",
        type=str,
        default="output_clips",
        help="Base directory for clip metadata"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache",
        help="Cache directory path"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_results",
        help="Output directory path"
    )
    parser.add_argument(
        "--num_neighbor",
        type=int,
        default=1,
        help="Number of neighboring clips to include (0 = none, 1 = one on each side, 2 = two on each side, default: 1)"
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default="http://127.0.0.1:22003/v1",
        help="API service URL (LLaVA-OneVision: 22004, Qwen3-VL: 22003)"
    )
    parser.add_argument(
        "--top_k_clips",
        type=int,
        default=4,
        help="Retrieve the top-K most relevant clips (default: 3)"
    )
    parser.add_argument(
        "--enable_rotation",
        action="store_true",
        default=False,
        help="Enable option-rotation evaluation (default: True; each question is rotated across A/B/C/D and is counted correct only if all four passes succeed)"
    )
    parser.add_argument(
        "--embedding_api_url",
        type=str,
        default="http://localhost:5000",
        help="Embedding API service URL (default: http://localhost:5000)"
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="",
        help="GPU ID suffix used to separate temporary output directories across processes (default: empty)"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    annotation_path = _resolve_path(args.annotation_path, project_root)
    clips_base_dir = _resolve_path(args.clips_base_dir, project_root)
    cache_dir = _resolve_path(args.cache_dir, project_root)
    output_path = _resolve_path(args.output_path, project_root)
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    video_path = annotations[0]['video_path']
    video_name = Path(video_path).stem
    
    clips_info_path = str((clips_base_dir / video_name / f"{video_name}_clips_info.json").resolve())
    
    inference_system = VideoQAInference(
        annotation_path=str(annotation_path),
        clips_info_path=clips_info_path,
        cache_dir=str(cache_dir),
        api_base_url=args.api_base_url,
        use_video_embedding=True,
        embedding_api_url=args.embedding_api_url,
        batch_size=4,
        output_dir=str((project_root / ".cache" / "qa_output" / (f"qa_{output_path.name}" + (f"_gpu{args.gpu_id}" if args.gpu_id else ""))).resolve()),
        clear_concept_db=False,
        num_neighbor=args.num_neighbor,
        enable_rotation=args.enable_rotation
    )

    inference_system.process_annotation_file(
        annotation_path=str(annotation_path),
        output_path=str(output_path),
        top_k_clips=args.top_k_clips
    )


if __name__ == "__main__":
    main()
