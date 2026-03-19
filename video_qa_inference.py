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
    remove_concept_markers, extract_question_without_options,
    evaluate_qa_results, print_evaluation_report,
    build_question_with_options, build_rotated_qa_item,
    has_complete_option_fields, extract_answer_from_response
)
from generate_concept_retrieval_descriptions import generate_distinctive_description


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
        model_path: str = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/OpenGVLab/InternVL3_5-8B",
        use_video_embedding: bool = False,
        embedding_api_url: str = "http://localhost:5000",
        batch_size: int = 10,
        output_dir: str = "./.cache/qa_output",
        clear_concept_db: bool = True,
        num_neighbor: int = 1,
        num_preceding_clips: int = 3,
        only_past_time_qa: bool = False,
        retrieve_for_current_time_qa: bool = True,
        replace_concept_in_query: bool = False,
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
            num_preceding_clips: 当前 clip 的前置 clip 数量（仅用于 past-time qa，默认：3）
            only_past_time_qa: 是否只评估 past-time qa（默认：False，会同时评估 past-time qa 和 current-time qa）
            retrieve_for_current_time_qa: current-time qa 是否检索历史片段（默认：True，False 时与 video_qa_inference_n.py 逻辑一致）
            replace_concept_in_query: 是否在检索时替换 query 中的概念名称为视觉描述（默认：False）
            enable_rotation: 是否启用选项轮换评估（默认：True）
        """
        print("=" * 80)
        print("初始化视频问答推理系统...")
        print("=" * 80)

        # 统一保存关键路径的绝对形式，避免依赖启动时 cwd
        self.annotation_path = str(Path(annotation_path).resolve())
        self.clips_info_path = str(Path(clips_info_path).resolve())
        self.annotation_base_dir = Path(self.annotation_path).parent
        
        # 保存邻居数量参数
        self.num_neighbor = num_neighbor
        print(f"邻居数量设置: {num_neighbor} (每个检索clip左右各添加 {num_neighbor} 个邻居)")
        
        # 保存前置 clip 数量参数
        self.num_preceding_clips = num_preceding_clips
        print(f"前置 clip 数量设置: {num_preceding_clips} (当前clip的前 {num_preceding_clips} 个clip)")
        
        # 保存只评估 past-time qa 的参数
        self.only_past_time_qa = only_past_time_qa
        qa_mode = "只评估 past-time qa" if only_past_time_qa else "评估 past-time qa 和 current-time qa"
        print(f"评估模式: {qa_mode}")
        
        # 保存 current-time qa 是否检索历史片段的参数
        self.retrieve_for_current_time_qa = retrieve_for_current_time_qa
        retrieve_mode = "检索历史片段" if retrieve_for_current_time_qa else "不检索历史片段"
        print(f"current-time qa 模式: {retrieve_mode}")
        
        # 保存是否替换 query 中概念名称的参数
        self.replace_concept_in_query = replace_concept_in_query
        concept_replace_mode = "替换为视觉描述" if replace_concept_in_query else "保留原始概念名称"
        print(f"检索 query 概念替换模式: {concept_replace_mode}")
        
        # 保存是否启用轮换评估参数
        self.enable_rotation = enable_rotation
        print(f"轮换评估模式: {'启用' if enable_rotation else '关闭'}")
        
        # 创建输出目录（使用绝对路径）
        self.output_dir = Path(output_dir).resolve()
        
        # 如果输出目录已存在，先删除
        if self.output_dir.exists():
            print(f"\n清理旧的输出目录: {self.output_dir}")
            shutil.rmtree(self.output_dir)
            print("✓ 旧输出目录已删除")
        
        # 重新创建输出目录
        self.output_dir.mkdir(exist_ok=True, parents=True)
        print(f"✓ 输出目录已创建: {self.output_dir}")
        
        # 初始化 ConceptDatabase
        print("\n[1/3] 初始化概念数据库...")
        
        # 根据 annotation 文件名创建专属的 cache 目录
        annotation_path_obj = Path(annotation_path)
        annotation_name = annotation_path_obj.stem  # 获取不带扩展名的文件名
        
        cache_dir_path = Path(cache_dir).resolve()
        annotation_cache_dir = cache_dir_path / annotation_name
        annotation_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置数据库路径和帧存储路径
        concept_db_path = annotation_cache_dir / "concept_db.json"
        frame_dir = annotation_cache_dir
        
        print(f"  Annotation: {annotation_name}")
        print(f"  数据库路径: {concept_db_path}")
        print(f"  帧存储路径: {frame_dir}")
        
        # 创建数据库实例
        self.concept_db = ConceptDatabase(db_path=str(concept_db_path), frame_dir=str(frame_dir))
        
        # 从 annotation 文件中提取并添加概念
        print(f"\n  从 annotation 文件提取概念定义...")
        concept_count = self.concept_db.add_concepts_from_annotation_file(
            annotation_file=self.annotation_path,
            clear_before_add=clear_concept_db
        )
        print(f"✓ 概念数据库初始化完成，共 {len(self.concept_db.data['concepts'])} 个概念")
        
        # 如果启用了概念替换，为每个概念生成 retrieval_description
        if self.replace_concept_in_query:
            print(f"\n  [概念替换] 正在为每个概念生成/更新 retrieval_description...")
            # 先初始化临时 client（后面会正式初始化）
            temp_client = OpenAI(api_key="EMPTY", base_url=api_base_url, timeout=3600)
            
            for concept in self.concept_db.data['concepts']:
                concept_name = concept.get('concept_name', 'Unknown')
                frame_path = concept.get('frame_path', '')
                original_description = concept.get('description', '')
                
                if not frame_path or not Path(frame_path).exists():
                    print(f"    ⚠ 概念 '{concept_name}' 的图像不存在，跳过")
                    continue
                
                print(f"    为概念 '{concept_name}' 生成 retrieval_description...")
                retrieval_desc = generate_distinctive_description(
                    client=temp_client,
                    model_path=model_path,
                    image_path=frame_path,
                    concept_name=concept_name,
                    original_description=original_description
                )
                concept['retrieval_description'] = retrieval_desc
                print(f"    ✓ {concept_name}: {retrieval_desc}")

            
            # 保存更新后的概念数据库
            self.concept_db._save_db()
            
            # 构建 concept_name -> retrieval_description 的映射
            self.concept_retrieval_map = {}
            for concept in self.concept_db.data['concepts']:
                name = concept.get('concept_name', '')
                desc = concept.get('retrieval_description', '')
                if name and desc:
                    self.concept_retrieval_map[name] = desc
            print(f"  ✓ 概念替换映射构建完成，共 {len(self.concept_retrieval_map)} 个映射")
        
        # 初始化 ClipMemory
        print("\n[2/3] 初始化 Clip 记忆系统...")
        embedding_type = "视频 Embedding" if use_video_embedding else "文本 Embedding"
        print(f"  检索模式: {embedding_type}")
        
        # 将 embeddings 缓存到与概念数据库相同的 annotation 专属目录
        embeddings_cache_dir = annotation_cache_dir
        print(f"  Embeddings 缓存目录: {embeddings_cache_dir}")
        
        self.clip_memory = ClipMemory(
            json_path=self.clips_info_path,
            api_base_url=embedding_api_url,
            use_video_embedding=use_video_embedding,
            batch_size=batch_size,
            cache_dir=str(embeddings_cache_dir),
            force_recompute=False  # 默认使用缓存
        )
        print(f"✓ Clip 记忆系统加载完成，共 {len(self.clip_memory.clips_data)} 个片段")
        
        # 初始化 OpenAI Client
        print("\n[3/3] 初始化模型 API...")
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_base_url,
            timeout=3600
        )
        self.model_path = model_path
        print(f"✓ 模型 API 初始化完成")
        
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

        # 兼容历史缓存中可能存在的相对路径
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
        # 提取 query 中的概念
        concepts_in_query = extract_concepts(query)
        if not concepts_in_query:
            return query
        
        # 检查是否有可用的替换映射
        replacements = {}
        for concept_name in concepts_in_query:
            if concept_name in self.concept_retrieval_map:
                replacements[concept_name] = self.concept_retrieval_map[concept_name]
        
        if not replacements:
            print(f"  ⚠ 未找到任何概念的 retrieval_description，使用原始 query")
            return query
        
        # 构建替换说明
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
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                max_tokens=512,
                temperature=0
            )
            rewritten_query = response.choices[0].message.content.strip()
            return rewritten_query
        except Exception as e:
            print(f"  ✗ 大模型重写 query 失败: {e}，回退到简单替换")
            # 回退：简单字符串替换
            result = query
            for name, desc in replacements.items():
                result = result.replace(f"{{{name}}}", desc)
            return result
    
    def retrieve_relevant_clips(
        self,
        question: str,
        max_time: str = None,
        top_k: int = 1,
        return_timing: bool = False
    ):
        """
        根据问题检索相关的视频片段
        
        Args:
            question: 问题文本
            max_time: 最大时间限制（格式: "HH:MM:SS"），只检索 end_time < max_time 的片段
            top_k: 返回前 k 个最相关的片段
            
        Returns:
            片段信息列表；
            当 return_timing=True 时，返回 (片段信息列表, 耗时统计字典)
        """
        import time as _time

        query_rewrite_time_ms = None
        top_k_retrieval_time_ms = None

        # 提取问题部分（不包含选项），保留概念标记用于更好的检索
        query = extract_question_without_options(question).strip()
        print(f"  原始 query: {query}")
        
        # 如果启用了概念替换，用大模型将 query 中的概念名称替换为视觉描述
        if self.replace_concept_in_query and hasattr(self, 'concept_retrieval_map') and self.concept_retrieval_map:
            _rewrite_t0 = _time.perf_counter()
            query = self.replace_concepts_with_descriptions(query)
            query_rewrite_time_ms = round((_time.perf_counter() - _rewrite_t0) * 1000)
            print(f"  替换后 query: {query}")
            print(f"  query rewrite 耗时: {query_rewrite_time_ms}ms")
        # import ipdb;ipdb.set_trace()
        print(f"******使用 {query} 检索相关视频片段")
        # 如果指定了时间限制，先过滤出符合条件的片段
        if max_time:
            max_seconds = time_to_seconds(max_time)
            
            # 临时保存原始数据
            original_clips_data = self.clip_memory.clips_data
            original_clip_embeddings = self.clip_memory.clip_embeddings
            
            # 过滤出 end_time < max_time 的片段
            filtered_indices = []
            filtered_clips = []
            for i, clip in enumerate(original_clips_data):
                if clip['end_time'] < max_seconds:
                    filtered_indices.append(i)
                    filtered_clips.append(clip)
            
            if not filtered_clips:
                print(f"  ⚠ 警告: 在时间 {max_time} 之前没有找到任何片段")
                if return_timing:
                    timing_stats = {
                        "query_rewrite_time_ms": query_rewrite_time_ms,
                        "top_k_retrieval_time_ms": top_k_retrieval_time_ms
                    }
                    return [], timing_stats
                return []
            
            # 临时替换数据
            self.clip_memory.clips_data = filtered_clips
            self.clip_memory.clip_embeddings = original_clip_embeddings[filtered_indices]
            
            print(f"  时间过滤: 从 {len(original_clips_data)} 个片段中筛选出 {len(filtered_clips)} 个（end_time < {max_time}）")
            
            # 执行搜索
            _retrieve_t0 = _time.perf_counter()
            results = self.clip_memory.search(query, top_k=top_k)
            top_k_retrieval_time_ms = round((_time.perf_counter() - _retrieve_t0) * 1000)
            
            # 恢复原始数据
            self.clip_memory.clips_data = original_clips_data
            self.clip_memory.clip_embeddings = original_clip_embeddings
        else:
            # 没有时间限制，直接搜索
            _retrieve_t0 = _time.perf_counter()
            results = self.clip_memory.search(query, top_k=top_k)
            top_k_retrieval_time_ms = round((_time.perf_counter() - _retrieve_t0) * 1000)

        if top_k_retrieval_time_ms is not None:
            print(f"  top-{top_k} clips 检索耗时: {top_k_retrieval_time_ms}ms")
        if return_timing:
            
            timing_stats = {
                "query_rewrite_time_ms": query_rewrite_time_ms,
                "top_k_retrieval_time_ms": top_k_retrieval_time_ms
            }
            return results, timing_stats
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
        
        # 如果 num_neighbor 为 0，直接返回原始检索结果（按时间排序并过滤）
        if self.num_neighbor == 0:
            print(f"\n  邻居数量为 0，不扩展邻居，直接使用检索结果")
            clips = [clip.copy() for clip in retrieved_clips]
            clips.sort(key=lambda x: x['start_time'])
            
            # 过滤重叠的片段
            if current_clip_start_time is not None:
                original_count = len(clips)
                clips = [clip for clip in clips if clip['end_time'] <= current_clip_start_time]
                filtered_count = original_count - len(clips)
                if filtered_count > 0:
                    print(f"  过滤掉 {filtered_count} 个与current clip重叠的片段")
            
            return clips
        
        # 构建 clip_id 到索引的映射
        clip_id_to_index = {}
        for i, clip in enumerate(self.clip_memory.clips_data):
            clip_id_to_index[clip['clip_id']] = i
        
        # 收集所有需要的clip id
        expanded_clip_ids = set()
        
        print(f"\n  扩展检索到的 {len(retrieved_clips)} 个clip，每个clip左右各添加 {self.num_neighbor} 个邻居...")
        
        for clip in retrieved_clips:
            clip_id = clip['clip_id']
            print(f"    检索到的 Clip ID: {clip_id}")
            
            # 添加当前clip
            expanded_clip_ids.add(clip_id)
            
            # 获取当前clip在原始列表中的索引
            if clip_id not in clip_id_to_index:
                print(f"      ⚠ 警告: Clip ID {clip_id} 在数据中未找到")
                continue
            
            current_index = clip_id_to_index[clip_id]
            
            # 添加前面 num_neighbor 个clip
            added_prev = []
            for offset in range(1, self.num_neighbor + 1):
                prev_index = current_index - offset
                if prev_index >= 0:
                    prev_clip = self.clip_memory.clips_data[prev_index]
                    prev_clip_id = prev_clip['clip_id']
                    expanded_clip_ids.add(prev_clip_id)
                    added_prev.append(prev_clip_id)
            
            if added_prev:
                print(f"      添加前邻居 Clip ID: {added_prev}")
            
            # 添加后面 num_neighbor 个clip
            added_next = []
            for offset in range(1, self.num_neighbor + 1):
                next_index = current_index + offset
                if next_index < len(self.clip_memory.clips_data):
                    next_clip = self.clip_memory.clips_data[next_index]
                    next_clip_id = next_clip['clip_id']
                    expanded_clip_ids.add(next_clip_id)
                    added_next.append(next_clip_id)
            
            if added_next:
                print(f"      添加后邻居 Clip ID: {added_next}")
        
        # 收集所有扩展后的clip
        expanded_clips = []
        for clip in self.clip_memory.clips_data:
            if clip['clip_id'] in expanded_clip_ids:
                expanded_clips.append(clip.copy())
        
        # 按开始时间排序
        expanded_clips.sort(key=lambda x: x['start_time'])
        
        # 如果指定了current_clip_start_time，过滤掉与current clip重叠的片段
        if current_clip_start_time is not None:
            original_count = len(expanded_clips)
            expanded_clips = [clip for clip in expanded_clips if clip['end_time'] <= current_clip_start_time]
            filtered_count = original_count - len(expanded_clips)
            if filtered_count > 0:
                print(f"  过滤掉 {filtered_count} 个与current clip重叠的片段")
        
        print(f"  扩展完成: {len(retrieved_clips)} 个检索clip → {len(expanded_clips)} 个总clip（包括邻居）")
        print(f"  最终clip ID列表: {[clip['clip_id'] for clip in expanded_clips]}")
        
        return expanded_clips
    
    def build_messages(
        self,
        question: str,
        concepts_info: List[Dict],
        clips_info: List[Dict],
        current_clip_path: Optional[str] = None,
        current_preceding_clips: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        构建发送给模型的 messages
        
        Args:
            question: 问题文本
            concepts_info: 概念信息列表
            clips_info: 历史相关片段信息列表
            current_clip_path: 当前视频片段路径（从片段开始到问题时间点）
            current_preceding_clips: 当前 clip 的前 n 个 clip（仅用于 past-time qa）
            
        Returns:
            messages 列表
        """
        content = []
        
        # 添加概念信息（图像或视频片段 + 文本描述）
        for concept in concepts_info:
            if concept is None:
                continue
            
            concept_type = concept.get('concept_type', 'frame')
            
            if concept_type == 'clip':
                # 片段模式：添加概念视频片段
                content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": _to_file_url(concept["frame_path"])
                    }
                })
            else:
                # 帧模式（原有）：添加概念图像
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": _to_file_url(concept["frame_path"])
                    }
                })
            
            # 添加概念说明文本
            content.append({
                "type": "text",
                "text": concept['description']
            })
        
        # 添加历史相关视频片段信息（视频 + 描述）
        if clips_info:
            content.append({
                "type": "text",
                "text": "Here are the previous related video clips:"
            })
            for clip in clips_info:
                # 添加视频片段
                content.append({
                    "type": "video_url",
                    "video_url": {
                        "url": _to_file_url(clip["clip_path"])
                    }
                })
                # if clip['description'] is not None and clip['description'].strip() != "":
                #     # 添加片段描述
                #     content.append({
                #         "type": "text",
                #         "text": f"{clip['description']}"
                #     })
        
        # # 添加当前 clip 的前 n 个 clip（仅用于 past-time qa）
        # if current_preceding_clips:
        #     content.append({
        #         "type": "text",
        #         "text": "Here are the clips immediately before the current clip:"
        #     })
        #     for clip in current_preceding_clips:
        #         content.append({
        #             "type": "video_url",
        #             "video_url": {
        #                 "url": f"file://{clip['clip_path']}"
        #             }
        #         })
        
        # 添加当前视频片段（从片段开始到问题时间点）
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

        # 添加问题
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
            (模型生成的回答, 纯模型推理耗时毫秒数, 非模型耗时毫秒数, call_model 总耗时毫秒数)
        """
        import time as _time
        _t0 = _time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            max_tokens=1024,
            temperature=0,
            # top_k=-1,
            # extra_body={"mm_processor_kwargs": {"fps": 1, "do_sample_frames": True}}
        )
        request_elapsed_ms = round((_time.perf_counter() - _t0) * 1000)
        answer = response.choices[0].message.content
        
        # 仅使用后端返回的纯模型推理耗时（仅 llm.generate 阶段）
        backend_inference_time_ms = None
        response_extra = getattr(response, "model_extra", None)
        if isinstance(response_extra, dict):
            backend_inference_time_ms = response_extra.get("inference_time_ms")
        
        if backend_inference_time_ms is None and hasattr(response, "model_dump"):
            response_dict = response.model_dump()
            if isinstance(response_dict, dict):
                backend_inference_time_ms = response_dict.get("inference_time_ms")
        
        non_inference_time_ms = None
        if backend_inference_time_ms is not None:
            non_inference_time_ms = request_elapsed_ms - backend_inference_time_ms
        
        return answer, backend_inference_time_ms, non_inference_time_ms, request_elapsed_ms
    
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
        
        # 1. 提取概念
        print("\n[步骤 1] 提取概念...")
        concepts = extract_concepts(question)
        print(f"  找到 {len(concepts)} 个概念: {concepts}")
        
        # 2. 从概念数据库检索概念信息
        print("\n[步骤 2] 检索概念信息...")
        concepts_info = []
        for concept_name in concepts:
            print(f"  检索概念: {concept_name}")
            concept_info = self.retrieve_concept_info(concept_name)
            if concept_info:
                print(f"    ✓ 找到: {concept_info['frame_path']}")
                concepts_info.append(concept_info)
        assert len(concepts)!=0, "问题中未提取到任何概念，请检查问题格式是否正确，概念应使用 {} 包围"
        # 3. 从 Clip 记忆系统检索相关片段（根据 qa_type 和 retrieve_for_current_time_qa 决定是否检索）
        clips_info = []
        # 支持两种格式：time（单帧格式）或 end_time（片段格式，回退到 end_time）
        question_time = qa_item.get('time') or qa_item.get('end_time')
        time_seconds = time_to_seconds(question_time)
        
        # 决定是否需要检索历史片段
        should_retrieve = (qa_type == 'past-time qa') or (qa_type == 'current-time qa' and self.retrieve_for_current_time_qa)
        query_rewrite_time_ms = None
        top_k_retrieval_time_ms = None
        
        if should_retrieve:
            print(f"\n[步骤 3] 检索相关视频片段 (Top {top_k_clips})...")
            retrieved_clips, retrieval_timing_stats = self.retrieve_relevant_clips(
                question,
                max_time=question_time,
                top_k=top_k_clips,
                return_timing=True
            )
            query_rewrite_time_ms = retrieval_timing_stats.get("query_rewrite_time_ms")
            top_k_retrieval_time_ms = retrieval_timing_stats.get("top_k_retrieval_time_ms")
            
            print(f"\n  检索到的top {len(retrieved_clips)} 个clip:")
            for i, clip in enumerate(retrieved_clips, 1):
                print(f"  #{i} Clip ID: {clip['clip_id']}, 相似度: {clip['similarity_score']:.4f}")
                print(f"      时间范围: {clip['start_time']:.2f}s - {clip['end_time']:.2f}s")
                print(f"      描述: {clip['description']}")
            
            # 获取当前时间点所在的片段，用于确定current clip的开始时间
            current_clip_info_temp = self.get_clip_at_time(time_seconds)
            if current_clip_info_temp is not None:
                current_clip_start_time = current_clip_info_temp['start_time']
            else:
                # 如果未找到对应片段，使用前1秒
                current_clip_start_time = max(0, time_seconds - 1)
            
            # 扩展clip列表，包括邻居
            clips_info = self.expand_clips_with_neighbors(retrieved_clips, current_clip_start_time)
            
            print(f"\n  扩展后的clip列表:")
            for i, clip in enumerate(clips_info, 1):
                print(f"  #{i} Clip ID: {clip['clip_id']}")
                print(f"      时间范围: {clip['start_time']:.2f}s - {clip['end_time']:.2f}s")
        else:
            print(f"\n[步骤 3] 跳过历史片段检索（current-time qa 且 retrieve_for_current_time_qa=False）")
        
        # 4. 提取当前视频片段（从片段开始到问题时间点）
        print("\n[步骤 4] 提取当前视频片段...")
        current_clip_path = None
        
        # 获取当前时间点所在的片段
        current_clip_info = self.get_clip_at_time(time_seconds)

        # 如果未找到对应片段，使用前1秒作为备选方案
        if current_clip_info is None:
            print(f"  ⚠ 在时间点 {question_time} 未找到对应的视频片段，使用前1秒作为当前片段")
            clip_start_time = max(0, time_seconds - 1)  # 确保不为负数
            clip_end_time = time_seconds
            clip_id_str = "fallback"
            is_fallback_clip = True
            print(f"  使用备选片段时间范围: {clip_start_time:.2f}s - {clip_end_time:.2f}s")
        else:
            clip_start_time = current_clip_info['start_time']
            clip_end_time = current_clip_info['end_time']
            clip_id_str = str(current_clip_info['clip_id'])
            is_fallback_clip = False
            print(f"  当前时间 {time_seconds:.2f}s 所在片段: Clip ID {current_clip_info['clip_id']}")
            print(f"  片段时间范围: {clip_start_time:.2f}s - {clip_end_time:.2f}s")
        
        # 计算片段时长，如果过短则延长
        clip_duration = time_seconds - clip_start_time
        adjusted_end_time = time_seconds
        
        # if clip_duration < 0.2:
        #     print(f"  ⚠ 片段时长过短 ({clip_duration:.2f}s < 0.2s)，将结束时间延长 0.5s")
        #     adjusted_end_time = time_seconds + 0.5
        #     # 确保不超过片段的实际结束时间
        #     if adjusted_end_time > clip_end_time:
        #         adjusted_end_time = clip_end_time
        #         print(f"    调整后的结束时间超过片段范围，限制为片段结束时间: {adjusted_end_time:.2f}s")
        #     else:
        #         print(f"    调整后的结束时间: {adjusted_end_time:.2f}s")
        
        # 提取从片段开始到调整后时间的视频
        source_video = self.clip_memory.source_video
        current_clip_filename = f"qa_{qa_id}_current_{clip_id_str}_{clip_start_time:.2f}_{adjusted_end_time:.2f}.mp4"
        current_clip_path = str((self.output_dir / current_clip_filename).resolve())
        
        print(f"  提取视频: {clip_start_time:.2f}s -> {adjusted_end_time:.2f}s (时长: {adjusted_end_time - clip_start_time:.2f}s)")
        success = extract_video_clip(
            source_video=source_video,
            start_time=clip_start_time,
            end_time=adjusted_end_time,
            output_path=current_clip_path,
            verbose=True
        )
        assert success
        # if not success:
        #     print(f"  ⚠ 警告: 提取当前视频片段失败，将不包含当前片段")
        #     current_clip_path = None

        # 4.5. 添加当前 clip 的前 n 个 clip（对 past-time qa 和 current-time qa 都适用）
        current_preceding_clips = []
        if self.num_preceding_clips > 0:
            
            print(f"\n[步骤 4.5] 添加当前 clip 的前 {self.num_preceding_clips} 个 clip...")
            n_preceding = self.num_preceding_clips
            
            # 收集已经存在的 clip id（来自检索的历史片段）
            existing_clip_ids = set(clip['clip_id'] for clip in clips_info)
            print(f"  已存在的历史 clip ID: {existing_clip_ids}")
            
            # 获取当前 clip 的索引
            current_clip_index = None
            if current_clip_info is not None:
                for i, clip in enumerate(self.clip_memory.clips_data):
                    if clip['clip_id'] == current_clip_info['clip_id']:
                        current_clip_index = i
                        break
            
            added_clips = []
            
            if current_clip_index is not None:
                # 情况1: 找到了当前 clip，从其索引往前查找
                print(f"  当前 clip 索引: {current_clip_index}")
                
                # 获取前 n 个 clip，跳过已存在的
                offset = 1
                while len(added_clips) < n_preceding and current_clip_index - offset >= 0:
                    preceding_clip = self.clip_memory.clips_data[current_clip_index - offset]
                    
                    # 如果这个 clip 不在已有的历史片段中，则添加
                    if preceding_clip['clip_id'] not in existing_clip_ids:
                        # 直接使用已有的 clip 路径
                        print(f"  添加前置 clip {len(added_clips)+1}: Clip ID {preceding_clip['clip_id']}")
                        print(f"    时间范围: {preceding_clip['start_time']:.2f}s - {preceding_clip['end_time']:.2f}s")
                        print(f"    路径: {preceding_clip['clip_path']}")
                        
                        added_clips.append({
                            'clip_id': preceding_clip['clip_id'],
                            'clip_path': preceding_clip['clip_path'],
                            'start_time': preceding_clip['start_time'],
                            'end_time': preceding_clip['end_time']
                        })
                    else:
                        print(f"  跳过 Clip ID {preceding_clip['clip_id']}（已存在于历史片段中）")
                    
                    offset += 1
            else:
                # 情况2: 未找到当前 clip（fallback情况），寻找距离问题时间点最近的前 n 个 clips
                print(f"  未找到当前 clip，寻找问题时间点 {time_seconds:.2f}s 之前距离最近的 {n_preceding} 个 clips...")
                
                # 找到所有 end_time <= time_seconds 的 clips
                candidate_clips = []
                for clip in self.clip_memory.clips_data:
                    if clip['end_time'] <= time_seconds and clip['clip_id'] not in existing_clip_ids:
                        candidate_clips.append(clip)
                
                # 按 end_time 降序排列（最近的在前）
                candidate_clips.sort(key=lambda x: x['end_time'], reverse=True)
                # 取前 n 个
                selected_clips = candidate_clips[:n_preceding]
                
                print(f"  找到 {len(candidate_clips)} 个候选 clips，选择最近的 {len(selected_clips)} 个")
                
                for clip in selected_clips:
                    # 直接使用已有的 clip 路径
                    print(f"  添加前置 clip {len(added_clips)+1}: Clip ID {clip['clip_id']}")
                    print(f"    时间范围: {clip['start_time']:.2f}s - {clip['end_time']:.2f}s")
                    print(f"    路径: {clip['clip_path']}")
                    
                    added_clips.append({
                        'clip_id': clip['clip_id'],
                        'clip_path': clip['clip_path'],
                        'start_time': clip['start_time'],
                        'end_time': clip['end_time']
                    })
            
            # 按时间顺序排列（从早到晚）
            added_clips.sort(key=lambda x: x['start_time'])
            current_preceding_clips = added_clips
            
            print(f"  ✓ 成功添加 {len(current_preceding_clips)} 个前置 clip")
        
        # 5. 构建 messages
        print("\n[步骤 5] 构建模型输入...")
        messages = self.build_messages(question, concepts_info, clips_info, current_clip_path, current_preceding_clips)
        print(f"  ✓ Messages 构建完成，共 {len(messages[0]['content'])} 个元素")
        if current_preceding_clips:
            print(f"  包含 {len(current_preceding_clips)} 个前置 clip")
        # if qa_type=="past-time qa":
        #     import ipdb;ipdb.set_trace()
        # 6. 调用模型（支持轮换评估）
        rotation_enabled_for_qa = (
            self.enable_rotation
            and has_complete_option_fields(qa_item)
            and str(qa_item.get("gt", "")).strip().upper() in ("A", "B", "C", "D")
        )
        original_gt = str(qa_item.get("gt", "")).strip().upper()
        # 
        # print("has_complete_option_fields(qa_item)",qa_item)
        # import ipdb;ipdb.set_trace()
        if rotation_enabled_for_qa:
            print("\n[步骤 6] 启用轮换评估，进行 4 次推理...")
            rotation_details = []
            all_correct = True
            total_inference_time_ms = 0
            total_non_inference_time_ms = 0
            total_call_model_time_ms = 0
            valid_inference_time_count = 0
            valid_non_inference_time_count = 0
            valid_call_model_time_count = 0
            for target_gt in ("A", "B", "C", "D"):
                rotated_qa = build_rotated_qa_item(qa_item, target_gt)
                rotated_question = build_question_with_options(
                    rotated_qa, require_complete_options=True
                )
                rotated_messages = self.build_messages(
                    rotated_question, concepts_info, clips_info, current_clip_path, current_preceding_clips
                )
                answer, call_time_ms, call_non_inference_time_ms, call_model_total_time_ms = self.call_model(rotated_messages)
                if call_time_ms is not None:
                    total_inference_time_ms += call_time_ms
                    valid_inference_time_count += 1
                if call_non_inference_time_ms is not None:
                    total_non_inference_time_ms += call_non_inference_time_ms
                    valid_non_inference_time_count += 1
                if call_model_total_time_ms is not None:
                    total_call_model_time_ms += call_model_total_time_ms
                    valid_call_model_time_count += 1
                predicted = extract_answer_from_response(answer).strip().upper()
                is_correct = predicted == target_gt
                all_correct = all_correct and is_correct
                rotation_details.append({
                    "target_gt": target_gt,
                    "rotated_question": rotated_question,
                    "predicted": predicted,
                    "is_correct": is_correct,
                    "raw_answer": answer,
                    "inference_time_ms": call_time_ms,
                    "non_inference_time_ms": call_non_inference_time_ms,
                    "call_model_time_ms": call_model_total_time_ms,
                })
                # import ipdb;ipdb.set_trace()
                print(
                    f"  轮换到 {target_gt}: predicted={predicted!r}, correct={is_correct}, "
                    f"call_model总耗时={call_model_total_time_ms}ms, "
                    f"推理耗时={call_time_ms}ms, 非模型耗时={call_non_inference_time_ms}ms"
                )

            print(f"  四次全部正确: {all_correct}")
            avg_inference_time_ms = (
                round(total_inference_time_ms / valid_inference_time_count)
                if valid_inference_time_count > 0 else None
            )
            avg_non_inference_time_ms = (
                round(total_non_inference_time_ms / valid_non_inference_time_count)
                if valid_non_inference_time_count > 0 else None
            )
            avg_call_model_time_ms = (
                round(total_call_model_time_ms / valid_call_model_time_count)
                if valid_call_model_time_count > 0 else None
            )
            print(f"  平均 call_model 总耗时: {avg_call_model_time_ms}ms")
            print(f"  平均推理耗时: {avg_inference_time_ms}ms")
            print(f"  平均非模型耗时: {avg_non_inference_time_ms}ms")
            final_answer = f"<ans>{original_gt}</ans>" if all_correct else ""
            inference_time_ms = avg_inference_time_ms
            non_inference_time_ms = avg_non_inference_time_ms
            call_model_time_ms = avg_call_model_time_ms
        else:
            print("\n[步骤 6] 调用模型进行推理...")
            answer, inference_time_ms, non_inference_time_ms, call_model_time_ms = self.call_model(messages)
            print(f"  ✓ 模型回答: {answer}")
            print(f"  ✓ call_model 总耗时: {call_model_time_ms}ms")
            print(f"  ✓ 推理耗时: {inference_time_ms}ms")
            print(f"  ✓ 非模型耗时: {non_inference_time_ms}ms")
            final_answer = answer
            all_correct = None
            rotation_details = []

        # 7. 添加 answer 字段和备选片段标记
        qa_item_with_answer = qa_item.copy()
        qa_item_with_answer['answer'] = final_answer
        qa_item_with_answer['is_fallback_clip'] = is_fallback_clip
        qa_item_with_answer['rotation_enabled'] = rotation_enabled_for_qa
        qa_item_with_answer['rotation_all_correct'] = all_correct
        qa_item_with_answer['rotation_details'] = rotation_details
        qa_item_with_answer['rotation_correct_flags'] = [
            item["is_correct"] for item in rotation_details
        ]
        qa_item_with_answer['call_model_time_ms'] = call_model_time_ms
        qa_item_with_answer['inference_time_ms'] = inference_time_ms
        qa_item_with_answer['non_inference_time_ms'] = non_inference_time_ms
        qa_item_with_answer['query_rewrite_time_ms'] = query_rewrite_time_ms
        qa_item_with_answer['top_k_retrieval_time_ms'] = top_k_retrieval_time_ms
        
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
        # 从 annotation_path 提取文件名并添加 _result 后缀
        annotation_filename = Path(annotation_path).stem  # 不带扩展名的文件名
        result_filename = f"{annotation_filename}_result.json"
        output_file_path = Path(output_path) / result_filename
        
        print(f"\n开始处理标注文件: {annotation_path}")
        print(f"输出文件: {output_file_path}\n")
        
        # 读取标注文件
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        results = []
        
        # 处理每个视频的问答
        for video_item in annotations:
            video_path = video_item['video_path']
            timestamps = video_item['timestamps']
            
            # 根据 only_past_time_qa 参数筛选问题
            if self.only_past_time_qa:
                # 只筛选 past-time qa
                target_qas = [qa for qa in timestamps if qa.get('qa_type') == 'past-time qa']
            else:
                # 筛选出 past-time qa 和 current-time qa 类型的问题
                target_qas = [qa for qa in timestamps if qa.get('qa_type') in ['past-time qa', 'current-time qa']]
            
            past_time_count = len([qa for qa in target_qas if qa.get('qa_type') == 'past-time qa'])
            current_time_count = len([qa for qa in target_qas if qa.get('qa_type') == 'current-time qa'])
            
            print(f"\n处理视频: {video_path}")
            print(f"总问题数: {len(timestamps)}")
            print(f"past-time qa 问题数: {past_time_count}")
            print(f"current-time qa 问题数: {current_time_count}")
            print(f"跳过其他类型问题: {len(timestamps) - len(target_qas)}")
            
            processed_timestamps = []
            
            for qa_item in target_qas:
                qa_with_answer = self.process_qa(qa_item, top_k_clips=top_k_clips)
                processed_timestamps.append(qa_with_answer)
            
            results.append({
                "video_path": video_path,
                "timestamps": processed_timestamps
            })
        
        # 保存结果
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ 处理完成！结果已保存到: {output_file_path}")
        print(f"{'='*80}\n")
        
        # 评估结果并打印报告
        eval_result = evaluate_qa_results(results)
        print_evaluation_report(eval_result)
        
        # 同时保存评估结果到 JSON 文件
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
        "--num_preceding_clips",
        type=int,
        default=0,
        help="当前 clip 的前置 clip 数量（仅用于 past-time qa，默认：3）"
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
        "--only_past_time_qa",
        action="store_true",
        help="是否只评估 past-time qa（默认：False，会同时评估 past-time qa 和 current-time qa）"
    )
    parser.add_argument(
        "--no_retrieve_for_current_time_qa",
        action="store_true",
        help="current-time qa 是否不检索历史片段（默认：False，即检索；设置此标志后不检索，与 video_qa_inference_n.py 逻辑一致）"
    )
    parser.add_argument(
        "--replace_concept_in_query",
        action="store_true",
        help="检索时是否将 query 中的概念名称替换为视觉描述（默认：False；启用后会为每个概念生成 retrieval_description 并用大模型重写 query）"
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
    # 解析命令行参数
    args = parse_args()

    # 统一以脚本所在目录（PEARL）作为相对路径基准，避免受启动 cwd 影响
    project_root = Path(__file__).resolve().parent
    annotation_path = _resolve_path(args.annotation_path, project_root)
    clips_base_dir = _resolve_path(args.clips_base_dir, project_root)
    cache_dir = _resolve_path(args.cache_dir, project_root)
    output_path = _resolve_path(args.output_path, project_root)
    
    # 读取 annotation 文件以获取 video_path
    print(f"\n读取 annotation 文件: {annotation_path}")
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    video_path = annotations[0]['video_path']
    print(f"从 annotation 文件中提取到 video_path: {video_path}")
    # 提取视频文件名（不含扩展名）
    video_name = Path(video_path).stem
    print(f"视频文件名: {video_name}")
    
    # 构建 clips_info_path
    clips_info_path = str((clips_base_dir / video_name / f"{video_name}_clips_info.json").resolve())
    print(f"自动推导的 clips_info_path: {clips_info_path}\n")
    
    # 初始化推理系统
    # 可以选择使用文本 Embedding 或视频 Embedding 进行检索
    inference_system = VideoQAInference(
        annotation_path=str(annotation_path),  # annotation 文件路径（用于提取概念定义）
        clips_info_path=clips_info_path,
        cache_dir=str(cache_dir),  # 缓存目录（自动创建子文件夹）
        api_base_url=args.api_base_url,  # API 服务地址
        use_video_embedding=True,  # True=视频 Embedding，False=文本 Embedding（默认）
        embedding_api_url=args.embedding_api_url,
        batch_size=4,  # 文本模式可以使用较大的批处理，视频模式建议 5-10
        output_dir=str((project_root / ".cache" / "qa_output" / (f"qa_{output_path.name}" + (f"_gpu{args.gpu_id}" if args.gpu_id else ""))).resolve()),  # 临时视频片段输出目录
        clear_concept_db=False,  # 每次运行时清空并重新构建概念数据库
        num_neighbor=args.num_neighbor,  # 邻居数量
        num_preceding_clips=args.num_preceding_clips,  # 前置 clip 数量
        only_past_time_qa=args.only_past_time_qa,  # 是否只评估 past-time qa
        retrieve_for_current_time_qa=not args.no_retrieve_for_current_time_qa,  # current-time qa 是否检索历史片段
        replace_concept_in_query=args.replace_concept_in_query,  # 检索时是否替换概念名称
        enable_rotation=args.enable_rotation
    )

    # 处理标注文件
    inference_system.process_annotation_file(
        annotation_path=str(annotation_path),
        output_path=str(output_path),
        top_k_clips=args.top_k_clips  # 检索最相关的 K 个片段
    )


if __name__ == "__main__":
    main()

