"""
视频问答推理脚本
整合 ConceptDatabase 和 ClipMemory，使用 Qwen3-VL 进行问答
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
    """将路径解析为绝对路径；相对路径基于 base_dir。"""
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    return (base_dir / p).resolve()


def _to_file_url(path_str: str) -> str:
    """将本地路径转换为标准 file:// URL。"""
    return Path(path_str).resolve().as_uri()


class VideoQAInference:
    """视频问答推理类"""
    
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
        初始化推理系统
        
        Args:
            annotation_path: annotation 文件路径（用于提取概念定义）
            clips_info_path: clips 信息 JSON 文件路径
            cache_dir: 缓存目录（用于存储概念数据库和帧图像）
            api_base_url: API 服务地址
            model_path: 模型路径
            use_video_embedding: 是否使用视频 Embedding 进行检索（True=视频，False=文本描述）
            embedding_api_url: Embedding API 服务地址
            batch_size: 预计算 Embedding 时的批处理大小
            output_dir: 输出目录（用于保存临时视频片段）
            clear_concept_db: 是否在添加概念前清空数据库（默认：True）
            num_neighbor: 邻居数量（0=不添加邻居，1=左右各1个，2=左右各2个，默认：1）
            enable_rotation: 是否启用选项轮换评估（默认：True）
        """
        print("=" * 80)
        print("初始化视频问答推理系统...")
        print("=" * 80)

        self.annotation_path = str(Path(annotation_path).resolve())
        self.clips_info_path = str(Path(clips_info_path).resolve())
        self.annotation_base_dir = Path(self.annotation_path).parent
        project_root = Path(__file__).resolve().parent
        model_path = str(_resolve_path(model_path, project_root))
        
        self.num_neighbor = num_neighbor
        self.enable_rotation = enable_rotation
        
        self.output_dir = Path(output_dir).resolve()
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print("\n[1/3] 初始化概念数据库...")

        annotation_path_obj = Path(annotation_path)
        annotation_name = annotation_path_obj.stem
        
        cache_dir_path = Path(cache_dir).resolve()
        annotation_cache_dir = cache_dir_path / annotation_name
        annotation_cache_dir.mkdir(parents=True, exist_ok=True)

        concept_db_path = annotation_cache_dir / "concept_db.json"
        frame_dir = annotation_cache_dir
        self.concept_db = ConceptDatabase(db_path=str(concept_db_path), frame_dir=str(frame_dir))

        self.concept_db.add_concepts_from_annotation_file(
            annotation_file=self.annotation_path,
            clear_before_add=clear_concept_db
        )
        print(f"✓ 概念数据库初始化完成，共 {len(self.concept_db.data['concepts'])} 个概念")
        
        temp_client = OpenAI(api_key="EMPTY", base_url=api_base_url, timeout=3600)
        
        for concept in self.concept_db.data['concepts']:
            concept_name = concept.get('concept_name', 'Unknown')
            frame_path = concept.get('frame_path', '')
            original_description = concept.get('description', '')
            retrieval_desc = generate_distinctive_description(
                client=temp_client,
                model_path=model_path,
                image_path=frame_path,
                concept_name=concept_name,
                original_description=original_description
            )
            concept['retrieval_description'] = retrieval_desc

        self.concept_db._save_db()
        self.concept_retrieval_map = {}
        for concept in self.concept_db.data['concepts']:
            name = concept.get('concept_name', '')
            desc = concept.get('retrieval_description', '')
            if name and desc:
                self.concept_retrieval_map[name] = desc

        print("\n[2/3] 初始化 Clip 记忆系统...")
        embeddings_cache_dir = annotation_cache_dir
        
        self.clip_memory = ClipMemory(
            json_path=self.clips_info_path,
            api_base_url=embedding_api_url,
            use_video_embedding=use_video_embedding,
            batch_size=batch_size,
            cache_dir=str(embeddings_cache_dir),
            force_recompute=False
        )
        print(f"✓ Clip 记忆系统加载完成，共 {len(self.clip_memory.clips_data)} 个片段")

        print("\n[3/3] 初始化模型 API...")
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_base_url,
            timeout=3600
        )
        self.model_path = model_path
        print(f"✓ 模型 API 初始化完成: {self.model_path}")
        
        print("\n" + "=" * 80)
        print("初始化完成！")
        print("=" * 80 + "\n")
    
    
    def retrieve_concept_info(self, concept_name: str) -> Dict:
        """
        从概念数据库检索概念信息
        
        Args:
            concept_name: 概念名称
            
        Returns:
            概念信息字典（包含 frame_path, concept_name 等）
        """
        concept_info = self.concept_db.query_by_name(concept_name)
        if concept_info is None:
            print(f"  ⚠ 警告: 概念 '{concept_name}' 在数据库中未找到")
            import ipdb;ipdb.set_trace()
            return None

        frame_path = concept_info.get("frame_path")
        if frame_path and not Path(frame_path).is_absolute():
            concept_info = concept_info.copy()
            concept_info["frame_path"] = str((Path(self.concept_db.db_path).parent / frame_path).resolve())
        return concept_info
    
    
    def get_clip_at_time(self, time_seconds: float) -> Optional[Dict]:
        """
        获取指定时间点所在的视频片段
        
        Args:
            time_seconds: 时间点（秒）
            
        Returns:
            片段信息字典，如果不存在则返回 None
        """
        for clip in self.clip_memory.clips_data:
            if clip['start_time'] <= time_seconds < clip['end_time']:
                return clip.copy()
        return None
    
    
    def replace_concepts_with_descriptions(self, query: str) -> str:
        """
        调用大模型将 query 中的 {ConceptName} 替换为对应的 retrieval_description，
        并保持语言流畅自然。
        
        Args:
            query: 包含 {ConceptName} 的原始 query
            
        Returns:
            替换后的 query（语言流畅）
        """
        concepts_in_query = extract_concepts(query)
        if not concepts_in_query:
            return query
        
        replacements = {}
        for concept_name in concepts_in_query:
            if concept_name in self.concept_retrieval_map:
                replacements[concept_name] = self.concept_retrieval_map[concept_name]
        
        if not replacements:
            print(f"  ⚠ 未找到任何概念的 retrieval_description，使用原始 query")
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
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
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
        根据问题检索相关的视频片段
        
        Args:
            question: 问题文本
            max_time: 最大时间限制（格式: "HH:MM:SS"），只检索 end_time < max_time 的片段
            top_k: 返回前 k 个最相关的片段
            
        Returns:
            片段信息列表
        """
        query = extract_question_without_options(question).strip()
        if self.concept_retrieval_map:
            query = self.replace_concepts_with_descriptions(query)
        if max_time:
            max_seconds = time_to_seconds(max_time)
            original_clips_data = self.clip_memory.clips_data
            original_clip_embeddings = self.clip_memory.clip_embeddings

            filtered_indices = []
            filtered_clips = []
            for i, clip in enumerate(original_clips_data):
                if clip['end_time'] < max_seconds:
                    filtered_indices.append(i)
                    filtered_clips.append(clip)
            
            if not filtered_clips:
                print(f"  ⚠ 警告: 在时间 {max_time} 之前没有找到任何片段")
                return []

            self.clip_memory.clips_data = filtered_clips
            self.clip_memory.clip_embeddings = original_clip_embeddings[filtered_indices]
            results = self.clip_memory.search(query, top_k=top_k)

            self.clip_memory.clips_data = original_clips_data
            self.clip_memory.clip_embeddings = original_clip_embeddings
        else:
            results = self.clip_memory.search(query, top_k=top_k)
        return results
    
    def expand_clips_with_neighbors(self, retrieved_clips: List[Dict], current_clip_start_time: float = None) -> List[Dict]:
        """
        扩展检索到的clip列表，包括每个clip的前后邻居
        
        Args:
            retrieved_clips: 检索到的clip列表
            current_clip_start_time: 当前clip的开始时间，用于避免重叠
            
        Returns:
            扩展后的clip列表（已去重并按时间排序）
        """
        if not retrieved_clips:
            return []
        
        if self.num_neighbor == 0:
            clips = [clip.copy() for clip in retrieved_clips]
            clips.sort(key=lambda x: x['start_time'])
            
            if current_clip_start_time is not None:
                original_count = len(clips)
                clips = [clip for clip in clips if clip['end_time'] <= current_clip_start_time]
                filtered_count = original_count - len(clips)
                if filtered_count > 0:
                    print(f"  过滤掉 {filtered_count} 个与current clip重叠的片段")
            
            return clips
        
        clip_id_to_index = {}
        for i, clip in enumerate(self.clip_memory.clips_data):
            clip_id_to_index[clip['clip_id']] = i
        
        expanded_clip_ids = set()
        
        for clip in retrieved_clips:
            clip_id = clip['clip_id']
            expanded_clip_ids.add(clip_id)

            if clip_id not in clip_id_to_index:
                print(f"      ⚠ 警告: Clip ID {clip_id} 在数据中未找到")
                continue
            
            current_index = clip_id_to_index[clip_id]
            for offset in range(1, self.num_neighbor + 1):
                prev_index = current_index - offset
                if prev_index >= 0:
                    prev_clip = self.clip_memory.clips_data[prev_index]
                    prev_clip_id = prev_clip['clip_id']
                    expanded_clip_ids.add(prev_clip_id)
            for offset in range(1, self.num_neighbor + 1):
                next_index = current_index + offset
                if next_index < len(self.clip_memory.clips_data):
                    next_clip = self.clip_memory.clips_data[next_index]
                    next_clip_id = next_clip['clip_id']
                    expanded_clip_ids.add(next_clip_id)

        expanded_clips = []
        for clip in self.clip_memory.clips_data:
            if clip['clip_id'] in expanded_clip_ids:
                expanded_clips.append(clip.copy())

        expanded_clips.sort(key=lambda x: x['start_time'])
        if current_clip_start_time is not None:
            original_count = len(expanded_clips)
            expanded_clips = [clip for clip in expanded_clips if clip['end_time'] <= current_clip_start_time]
            filtered_count = original_count - len(expanded_clips)
            if filtered_count > 0:
                print(f"  过滤掉 {filtered_count} 个与current clip重叠的片段")
        
        return expanded_clips
    
    def build_messages(
        self,
        question: str,
        concepts_info: List[Dict],
        clips_info: List[Dict],
        current_clip_path: Optional[str] = None
    ) -> List[Dict]:
        """
        构建发送给模型的 messages
        
        Args:
            question: 问题文本
            concepts_info: 概念信息列表
            clips_info: 历史相关片段信息列表
            current_clip_path: 当前视频片段路径（从片段开始到问题时间点）
            
        Returns:
            messages 列表
        """
        content = []
        for concept in concepts_info:
            if concept is None:
                continue
            
            concept_type = concept.get('concept_type', 'frame')
            
            if concept_type == 'clip':
                content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": _to_file_url(concept["frame_path"])
                    }
                })
            else:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": _to_file_url(concept["frame_path"])
                    }
                })
            
            content.append({
                "type": "text",
                "text": concept['description']
            })
        
        if clips_info:
            content.append({
                "type": "text",
                "text": "Here are the previous related video clips:"
            })
            for clip in clips_info:
                content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": _to_file_url(clip["clip_path"])
                    }
                })
        
        if current_clip_path:
            content.append({
                "type": "text",
                "text": "The following is the current video clip:"
            })
            content.append({
                "type": "video_url",
                "video_url": {
                    "url": _to_file_url(current_clip_path)
                }
            })

        content.append({
            "type": "text",
            "text": f"{question}\n\nPlease output your answer choice in <ans></ans> tags."
        })
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def call_model(self, messages: List[Dict]) -> tuple:
        """
        调用模型进行推理
        
        Args:
            messages: 消息列表
            
        Returns:
            模型生成的回答
        """
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            max_tokens=1024,
            temperature=0
        )
        answer = response.choices[0].message.content
        return answer
    
    def process_qa(self, qa_item: Dict, top_k_clips: int = 1) -> Dict:
        """
        处理单个问答项
        
        Args:
            qa_item: 问答项字典
            top_k_clips: 检索的片段数量
            
        Returns:
            添加了 answer 字段的问答项
        """
        question = build_question_with_options(qa_item, require_complete_options=False)
        qa_id = qa_item['id']
        qa_type = qa_item['qa_type']
        
        print(f"\n{'='*80}")
        print(f"处理问题 ID: {qa_id}")
        print(f"问题类型: {qa_type}")
        print(f"时间: {qa_item.get('time') or qa_item.get('end_time')}")
        print(f"问题: {question}")
        print(f"{'='*80}")

        print("\n[步骤 1] 提取概念...")
        concepts = extract_concepts(question)

        print("\n[步骤 2] 检索概念信息...")
        concepts_info = []
        for concept_name in concepts:
            concept_info = self.retrieve_concept_info(concept_name)
            if concept_info:
                concepts_info.append(concept_info)
        assert len(concepts)!=0, "问题中未提取到任何概念，请检查问题格式是否正确，概念应使用 {} 包围"
        clips_info = []
        question_time = qa_item.get('time') or qa_item.get('end_time')
        time_seconds = time_to_seconds(question_time)

        print(f"\n[步骤 3] 检索相关视频片段 (Top {top_k_clips})...")
        retrieved_clips = self.retrieve_relevant_clips(
            question,
            max_time=question_time,
            top_k=top_k_clips
        )

        current_clip_info_temp = self.get_clip_at_time(time_seconds)
        if current_clip_info_temp is not None:
            current_clip_start_time = current_clip_info_temp['start_time']
        else:
            current_clip_start_time = max(0, time_seconds - 1)

        clips_info = self.expand_clips_with_neighbors(retrieved_clips, current_clip_start_time)

        print("\n[步骤 4] 提取当前视频片段...")
        current_clip_path = None
        current_clip_info = self.get_clip_at_time(time_seconds)

        if current_clip_info is None:
            print(f"  ⚠ 在时间点 {question_time} 未找到对应的视频片段，使用前1秒作为当前片段")
            clip_start_time = max(0, time_seconds - 1)
            clip_id_str = "fallback"
            print(f"  使用备选片段时间范围: {clip_start_time:.2f}s - {time_seconds:.2f}s")
        else:
            clip_start_time = current_clip_info['start_time']
            clip_id_str = str(current_clip_info['clip_id'])

        source_video = self.clip_memory.source_video
        current_clip_filename = f"qa_{qa_id}_current_{clip_id_str}_{clip_start_time:.2f}_{time_seconds:.2f}.mp4"
        current_clip_path = str((self.output_dir / current_clip_filename).resolve())
        success = extract_video_clip(
            source_video=source_video,
            start_time=clip_start_time,
            end_time=time_seconds,
            output_path=current_clip_path,
            verbose=True
        )
        assert success

        print("\n[步骤 5] 构建模型输入...")
        messages = self.build_messages(question, concepts_info, clips_info, current_clip_path)
        rotation_enabled_for_qa = (
            self.enable_rotation
            and has_complete_option_fields(qa_item)
            and str(qa_item.get("gt", "")).strip().upper() in ("A", "B", "C", "D")
        )
        original_gt = str(qa_item.get("gt", "")).strip().upper()
        if rotation_enabled_for_qa:
            print("\n[步骤 6] 启用轮换评估，进行 4 次推理...")
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

            print(f"  四次全部正确: {all_correct}")
            final_answer = f"<ans>{original_gt}</ans>" if all_correct else ""
        else:
            print("\n[步骤 6] 调用模型进行推理...")
            answer = self.call_model(messages)
            print(f"  ✓ 模型回答: {answer}")
            final_answer = answer
            all_correct = None
            rotation_details = []

        qa_item_with_answer = qa_item.copy()
        qa_item_with_answer['answer'] = final_answer
        qa_item_with_answer['rotation_enabled'] = rotation_enabled_for_qa
        qa_item_with_answer['rotation_all_correct'] = all_correct
        qa_item_with_answer['rotation_details'] = rotation_details
        qa_item_with_answer['rotation_correct_flags'] = [
            item["is_correct"] for item in rotation_details
        ]
        
        return qa_item_with_answer
    
    def process_annotation_file(
        self,
        annotation_path: str,
        output_path: str,
        top_k_clips: int = 1
    ):
        """
        处理整个标注文件
        
        Args:
            annotation_path: 输入标注文件路径
            output_path: 输出结果文件夹路径
            top_k_clips: 检索的片段数量
        """
        annotation_filename = Path(annotation_path).stem
        result_filename = f"{annotation_filename}_result.json"
        output_file_path = Path(output_path) / result_filename
        
        print(f"\n开始处理标注文件: {annotation_path}")
        print(f"输出文件: {output_file_path}\n")

        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        results = []
        for video_item in annotations:
            video_path = video_item['video_path']
            timestamps = video_item['timestamps']
            target_qas = [qa for qa in timestamps if qa.get('qa_type') in ['past-time qa', 'current-time qa']]
            
            past_time_count = len([qa for qa in target_qas if qa.get('qa_type') == 'past-time qa'])
            current_time_count = len([qa for qa in target_qas if qa.get('qa_type') == 'current-time qa'])
            
            print(f"\n处理视频: {video_path}")
            print(f"past-time qa 问题数: {past_time_count}")
            print(f"current-time qa 问题数: {current_time_count}")
            
            processed_timestamps = []
            
            for qa_item in target_qas:
                qa_with_answer = self.process_qa(qa_item, top_k_clips=top_k_clips)
                processed_timestamps.append(qa_with_answer)
            
            results.append({
                "video_path": video_path,
                "timestamps": processed_timestamps
            })

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ 处理完成！结果已保存到: {output_file_path}")
        print(f"{'='*80}\n")

        eval_result = evaluate_qa_results(results)
        print_evaluation_report(eval_result)

        eval_filename = f"{annotation_filename}_evaluation.json"
        eval_file_path = Path(output_path) / eval_filename
        with open(eval_file_path, 'w', encoding='utf-8') as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)
        print(f"\n✓ 评估结果已保存到: {eval_file_path}\n")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="视频问答推理脚本")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="data/annotations/thief.json",
        help="标注文件路径"
    )
    parser.add_argument(
        "--clips_base_dir",
        type=str,
        default="output_clips",
        help="片段信息基础目录路径"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=".cache",
        help="缓存目录路径"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output_results",
        help="输出文件夹路径"
    )
    parser.add_argument(
        "--num_neighbor",
        type=int,
        default=1,
        help="邻居数量（0=不添加邻居，1=左右各1个，2=左右各2个，默认：1）"
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default="http://127.0.0.1:22003/v1",
        help="API 服务地址（LLaVA-OneVision: 22004, Qwen3-VL: 22003）"
    )
    parser.add_argument(
        "--top_k_clips",
        type=int,
        default=4,
        help="检索最相关的 K 个片段（默认：3）"
    )
    parser.add_argument(
        "--enable_rotation",
        action="store_true",
        default=False,
        help="是否启用选项轮换评估（默认：True，每题轮换 A/B/C/D 共4次，四次全对才判对）"
    )
    parser.add_argument(
        "--embedding_api_url",
        type=str,
        default="http://localhost:5000",
        help="Embedding API 服务地址（默认：http://localhost:5000）"
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="",
        help="GPU ID 后缀，用于区分不同进程的临时输出目录，避免冲突（默认：空）"
    )
    return parser.parse_args()

def main():
    """主函数"""
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
