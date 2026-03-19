import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict
import subprocess
from utils import extract_video_frame


class ConceptDatabase:
    """
    简单的概念数据库，用于存储和查询视频中定义的概念
    
    每个概念包含：
    - concept_name: 概念名称
    - description: 模型生成的简短描述
    - frame_path: 当前帧图像的存储路径（自动从视频提取）
    - timestamp: 时间戳
    - video_path: 视频路径
    
    特性：
    - 自动从视频中提取指定时间戳的帧
    - 使用 concept_name 命名保存的帧图像
    - 支持增加和查询功能
    """
    
    def __init__(self, db_path: str = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache/concept_db.json", frame_dir: str = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache"):
        """
        初始化数据库
        
        Args:
            db_path: 数据库JSON文件路径（默认：/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0117/.cache/concept_db.json）
            frame_dir: 存储概念帧图像的目录（默认：/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0117/.cache）
        """
        self.db_path = Path(db_path)
        self.frame_dir = Path(frame_dir)
        
        # 创建帧存储目录
        self.frame_dir.mkdir(exist_ok=True)
        
        # 加载或初始化数据库
        self.data = self._load_db()
    
    def _load_db(self) -> Dict:
        """加载数据库文件"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"警告: 数据库文件 {self.db_path} 格式错误，将创建新数据库")
                return {"concepts": []}
        else:
            return {"concepts": []}
    
    def _save_db(self):
        """保存数据库到文件"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def _extract_frame(self, video_path: str, timestamp: str, output_path: str) -> bool:
        """
        从视频中提取指定时间戳的帧
        
        Args:
            video_path: 视频文件路径
            timestamp: 时间戳 (格式: HH:MM:SS)
            output_path: 输出图像路径
            
        Returns:
            bool: 提取成功返回True，否则返回False
        """
        return extract_video_frame(
            source_video=video_path,
            timestamp=timestamp,
            output_path=output_path,
            verbose=True
        )

    def _extract_clip(self, video_path: str, start_time: str, end_time: str, output_path: str) -> bool:
        """
        从视频中提取指定时间段的视频片段
        
        Args:
            video_path: 视频文件路径
            start_time: 开始时间戳 (格式: HH:MM:SS)
            end_time: 结束时间戳 (格式: HH:MM:SS)
            output_path: 输出视频路径
            
        Returns:
            bool: 提取成功返回True，否则返回False
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', start_time,
                '-to', end_time,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',  # 覆盖已存在的文件
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True
            )
            
            if os.path.exists(output_path):
                print(f"✓ 成功提取片段: {output_path}")
                return True
            else:
                print(f"✗ 提取片段失败: 输出文件不存在")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"✗ ffmpeg 提取片段失败: {e}")
            print(f"stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"✗ 提取片段异常: {e}")
            return False
    
    def add_concept(
        self,
        concept_name: str,
        description: str,
        video_path: str,
        timestamp: str = None,
        start_time: str = None,
        end_time: str = None,
        additional_info: Optional[Dict] = None
    ) -> bool:
        """
        添加一个新概念到数据库
        
        支持两种模式：
        - 帧模式（原有）：传入 timestamp，从视频中提取单帧图像
        - 片段模式（新增）：传入 start_time + end_time，从视频中提取视频片段
        
        Args:
            concept_name: 概念名称
            description: 模型生成的描述
            video_path: 视频路径
            timestamp: 时间戳 (格式: HH:MM:SS)，帧模式时使用
            start_time: 片段开始时间 (格式: HH:MM:SS)，片段模式时使用
            end_time: 片段结束时间 (格式: HH:MM:SS)，片段模式时使用
            additional_info: 额外的信息（可选）
            
        Returns:
            bool: 添加成功返回True，否则返回False
        """
        try:
            if start_time and end_time:
                # 片段模式：提取视频片段
                clip_filename = f"{concept_name}.mp4"
                target_path = self.frame_dir / clip_filename
                
                if not self._extract_clip(video_path, start_time, end_time, str(target_path)):
                    print(f"✗ 无法从视频提取片段，跳过添加概念 {concept_name}")
                    return False
                
                # 创建概念记录（concept_type = "clip"）
                concept_entry = {
                    "concept_name": concept_name,
                    "description": description,
                    "frame_path": str(target_path),  # 保持向后兼容，存储片段路径
                    "concept_type": "clip",
                    "start_time": start_time,
                    "end_time": end_time,
                    "video_path": video_path,
                }
            else:
                # 帧模式（原有逻辑）：提取单帧图像
                frame_filename = f"{concept_name}.jpg"
                target_frame_path = self.frame_dir / frame_filename
                
                if not self._extract_frame(video_path, timestamp, str(target_frame_path)):
                    print(f"✗ 无法从视频提取帧，跳过添加概念 {concept_name}")
                    return False
                
                # 创建概念记录（concept_type = "frame"）
                concept_entry = {
                    "concept_name": concept_name,
                    "description": description,
                    "frame_path": str(target_frame_path),
                    "concept_type": "frame",
                    "timestamp": timestamp,
                    "video_path": video_path,
                }
            
            # 添加额外信息
            if additional_info:
                concept_entry.update(additional_info)
            
            # 添加到数据库
            self.data["concepts"].append(concept_entry)
            
            # 保存到文件
            self._save_db()
            
            print(f"✓ 成功添加概念: {concept_name} (类型: {concept_entry['concept_type']})")
            return True
            
        except Exception as e:
            print(f"✗ 添加概念 {concept_name} 失败: {e}")
            return False
    
    def query_by_name(self, concept_name: str) -> Optional[Dict]:
        """
        根据概念名称查询
        
        Args:
            concept_name: 概念名称
            
        Returns:
            Dict: 概念信息，如果不存在返回None
        """
        for concept in self.data["concepts"]:
            if concept["concept_name"] == concept_name:
                return concept
        return None
    
    def clear_database(self):
        """清空数据库中的所有概念"""
        self.data = {"concepts": []}
        self._save_db()
        print("✓ 数据库已清空")
    
    def extract_concept_names(self, text: str) -> List[str]:
        """
        从文本中提取所有被 {} 包围的概念名称
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 提取的概念名称列表
        """
        # 使用正则表达式匹配 {concept_name} 格式
        pattern = r'\{([^}]+)\}'
        concepts = re.findall(pattern, text)
        return concepts
    
    def add_concepts_from_annotation_file(
        self, 
        annotation_file: str,
        clear_before_add: bool = True
    ) -> int:
        """
        从 annotation 文件中批量添加概念
        
        该方法会：
        1. 读取 annotation 文件
        2. 遍历所有 concept definition 类型的条目
        3. 提取概念名称并添加到数据库
        4. 跳过重复的概念
        
        Args:
            annotation_file: annotation 文件路径
            clear_before_add: 是否在添加前清空数据库（默认：True）
            
        Returns:
            int: 成功添加的概念数量
        """
        # 1. 清空数据库（如果需要）
        if clear_before_add:
            print("\n清空数据库...")
            self.clear_database()
        
        annotation_file_path = Path(annotation_file).resolve()
        project_root = Path(__file__).resolve().parent

        # 2. 读取 annotation 文件
        print(f"\n读取标注文件: {annotation_file}")
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"✓ 成功加载标注文件")
        
        # 3. 提取并添加概念定义
        print("\n提取并添加概念定义...")
        concept_count = 0
        skipped_count = 0
        error_count = 0
        
        for video_item in annotations:
            video_path_raw = video_item.get('video_path')
            if not video_path_raw:
                raise ValueError("annotation 中存在缺失 video_path 的条目")

            video_path_obj = Path(video_path_raw)
            if not video_path_obj.is_absolute():
                video_path_obj = (project_root / video_path_obj).resolve()
            else:
                video_path_obj = video_path_obj.resolve()

            if not video_path_obj.exists():
                raise FileNotFoundError(f"视频文件不存在: {video_path_obj}")

            video_path = str(video_path_obj)
            timestamps = video_item.get('timestamps', [])
            
            print(f"\n处理视频: {video_path}")
            
            for qa_item in timestamps:
                # 只处理 concept definition 类型
                if qa_item.get('qa_type') != "concept definition":
                    continue
                
                question = qa_item.get('question', '')
                time = qa_item.get('time', '')
                start_time = qa_item.get('start_time', '')
                end_time = qa_item.get('end_time', '')
                
                # 提取概念名称
                concept_names = self.extract_concept_names(question)
                
                if not concept_names:
                    print(f"  ⚠ 警告: 在问题中未找到概念名称: {question}")
                    continue
                
                # 判断模式：优先使用 start_time/end_time（片段模式），否则用 time（帧模式）
                use_clip_mode = bool(start_time and end_time)
                if use_clip_mode:
                    print(f"  [片段模式] 时间范围: {start_time} -> {end_time}")
                else:
                    print(f"  [帧模式] 时间戳: {time}")
                
                # 添加每个概念
                for concept_name in concept_names:
                    # 检查概念是否已存在
                    existing_concept = self.query_by_name(concept_name)
                    if existing_concept:
                        print(f"  ⊗ 跳过重复概念: {concept_name} (已存在)")
                        skipped_count += 1
                        continue
                    
                    print(f"\n  添加概念: {concept_name}")
                    print(f"    问题: {question[:80]}...")
                    
                    success = self.add_concept(
                        concept_name=concept_name,
                        description=question,
                        video_path=video_path,
                        timestamp=time if not use_clip_mode else None,
                        start_time=start_time if use_clip_mode else None,
                        end_time=end_time if use_clip_mode else None
                    )
                    
                    if success:
                        concept_count += 1
                    else:
                        error_count += 1
        
        # 4. 显示统计信息
        print("\n" + "-" * 80)
        print(f"✓ 处理完成！")
        print(f"  成功添加: {concept_count} 个概念")
        print(f"  跳过重复: {skipped_count} 个概念")
        print(f"  添加失败: {error_count} 个概念")
        assert error_count==0, "添加概念时发生错误，请检查日志"
        print("-" * 80)
        
        return concept_count


# 使用示例
if __name__ == "__main__":
    print("=" * 80)
    print("开始处理概念定义")
    print("=" * 80)
    
    # 配置路径
    base_cache_dir = Path("/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache")
    annotation_file = Path("/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/annotations/apartment5.json")
    
    # 检查文件是否存在
    if not annotation_file.exists():
        print(f"✗ 文件不存在: {annotation_file}")
        exit(1)
    
    annotation_name = annotation_file.stem  # 获取不带扩展名的文件名
    
    print(f"\n处理 annotation: {annotation_file.name}")
    print("=" * 80)
    
    # 为当前 annotation 创建专属的子文件夹
    annotation_cache_dir = base_cache_dir / annotation_name
    annotation_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置数据库路径和帧存储路径
    db_path = annotation_cache_dir / "concept_db.json"
    frame_dir = annotation_cache_dir
    
    print(f"\n数据库路径: {db_path}")
    print(f"帧存储路径: {frame_dir}")
    
    # 创建数据库实例
    db = ConceptDatabase(db_path=str(db_path), frame_dir=str(frame_dir))
    
    # 从 annotation 文件批量添加概念
    concept_count = db.add_concepts_from_annotation_file(
        annotation_file=str(annotation_file),
        clear_before_add=True
    )
    
    # 显示当前 annotation 的所有概念
    if concept_count > 0:
        print(f"\n{annotation_name} 数据库中的所有概念:")
        print("-" * 80)
        for i, concept in enumerate(db.data['concepts'], 1):
            print(f"{i}. {concept['concept_name']}")
            print(f"   时间戳: {concept['timestamp']}")
            print(f"   帧路径: {concept['frame_path']}")
            print()
    
    print("\n" + "=" * 80)
    print(f"✓ 处理完成！共添加 {concept_count} 个概念")
    print("=" * 80)
