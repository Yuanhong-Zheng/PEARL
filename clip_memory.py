"""
Memory System - 视频片段记忆系统
使用 Qwen3-VL-Embedding API 进行语义相似度检索
"""
import json
import requests
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import hashlib


class ClipMemory:
    """视频片段记忆类"""
    
    def __init__(self, json_path: str, api_base_url: str = "http://localhost:5000", 
                 use_video_embedding: bool = False, batch_size: int = 10,
                 cache_dir: str = None, force_recompute: bool = False):
        """
        初始化记忆系统
        
        Args:
            json_path: clips 信息 JSON 文件路径
            api_base_url: Embedding API 服务基础地址
            use_video_embedding: 是否使用视频 Embedding 进行检索（True=使用视频，False=使用文本描述）
            batch_size: 预计算时的批处理大小，视频模式建议 5-10，文本模式可以更大（如 50-100）
            cache_dir: 缓存目录，如果为 None，则使用 json_path 所在目录
            force_recompute: 是否强制重新计算 embeddings（忽略缓存）
        """
        self.json_path = json_path
        self.api_base_url = api_base_url
        self.use_video_embedding = use_video_embedding
        self.clips_data = []
        self.source_video = ""
        self.clip_embeddings = None  # 存储预计算的 embeddings
        self.force_recompute = force_recompute
        
        # 设置缓存目录
        if cache_dir is None:
            # 默认使用 json_path 所在目录
            cache_dir = str(Path(json_path).parent)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self._load_clips()
        
        # 预计算所有 clip 的 embeddings（带缓存功能）
        self._precompute_embeddings(batch_size=batch_size)
        
    def _load_clips(self):
        """从 JSON 文件加载片段信息"""
        print(f"正在加载片段数据: {self.json_path}")
        
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        project_root = Path(__file__).resolve().parent

        self.source_video = data.get("source_video", "")
        if self.source_video:
            source_video_path = Path(self.source_video)
            if not source_video_path.is_absolute():
                source_video_path = (project_root / source_video_path).resolve()
            else:
                source_video_path = source_video_path.resolve()
            self.source_video = str(source_video_path)
        clips = data.get("clips", [])
        
        # 提取所需字段
        for clip in clips:
            clip_path = clip.get("clip_path")
            if clip_path:
                clip_path_obj = Path(clip_path)
                if not clip_path_obj.is_absolute():
                    clip_path_obj = (project_root / clip_path_obj).resolve()
                else:
                    clip_path_obj = clip_path_obj.resolve()
                clip_path = str(clip_path_obj)

            clip_info = {
                "clip_id": clip.get("clip_id"),
                "clip_path": clip_path,
                "start_time": clip.get("start_time"),
                "end_time": clip.get("end_time"),
                "duration": clip.get("duration"),
                "description": clip.get("description", "")
            }
            self.clips_data.append(clip_info)
        
        print(f"成功加载 {len(self.clips_data)} 个视频片段")
    
    def _get_cache_path(self) -> Path:
        """
        生成缓存文件路径
        缓存文件名基于 json_path 和 embedding 类型
        
        Returns:
            缓存文件的 Path 对象
        """
        # 获取 json 文件名（不含扩展名）
        json_filename = Path(self.json_path).stem
        
        # 根据 embedding 类型添加后缀
        embedding_suffix = "video" if self.use_video_embedding else "text"
        
        # 生成缓存文件名
        cache_filename = f"{json_filename}_embeddings_{embedding_suffix}.npy"
        
        return self.cache_dir / cache_filename
    
    def _get_clips_hash(self) -> str:
        """
        计算 clips 数据的哈希值，用于验证缓存有效性
        
        Returns:
            clips 数据的 MD5 哈希值
        """
        # 提取关键信息（clip_id, clip_path, description）
        clips_key_info = []
        for clip in self.clips_data:
            if self.use_video_embedding:
                # 视频模式：使用 clip_path
                key = f"{clip['clip_id']}:{clip['clip_path']}"
            else:
                # 文本模式：使用 description
                key = f"{clip['clip_id']}:{clip['description']}"
            clips_key_info.append(key)
        
        # 计算哈希
        clips_str = "||".join(clips_key_info)
        hash_obj = hashlib.md5(clips_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _save_embeddings_cache(self, embeddings: np.ndarray):
        """
        保存 embeddings 到缓存文件
        
        Args:
            embeddings: 要保存的 embeddings 数组
        """
        cache_path = self._get_cache_path()
        clips_hash = self._get_clips_hash()
        
        # 保存 embeddings 和元数据
        cache_data = {
            'embeddings': embeddings,
            'clips_hash': clips_hash,
            'num_clips': len(self.clips_data),
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])
        }
        
        np.save(cache_path, cache_data, allow_pickle=True)
        print(f"✓ Embeddings 已缓存到: {cache_path}")
    
    def _load_embeddings_cache(self) -> Optional[np.ndarray]:
        """
        从缓存文件加载 embeddings
        
        Returns:
            加载的 embeddings 数组，如果缓存无效则返回 None
        """
        cache_path = self._get_cache_path()
        
        # 检查缓存文件是否存在
        if not cache_path.exists():
            print(f"未找到缓存文件: {cache_path}")
            return None
        
        try:
            # 加载缓存数据
            cache_data = np.load(cache_path, allow_pickle=True).item()
            
            # 验证缓存有效性
            cached_hash = cache_data.get('clips_hash', '')
            current_hash = self._get_clips_hash()
            
            if cached_hash != current_hash:
                print(f"⚠ 缓存已过期（clips 数据已更改），将重新计算")
                return None
            
            cached_num_clips = cache_data.get('num_clips', 0)
            if cached_num_clips != len(self.clips_data):
                print(f"⚠ 缓存片段数量不匹配（缓存: {cached_num_clips}, 当前: {len(self.clips_data)}），将重新计算")
                return None
            
            embeddings = cache_data['embeddings']
            print(f"✓ 从缓存加载 embeddings: {cache_path}")
            print(f"  形状: {embeddings.shape}, 片段数: {len(self.clips_data)}")
            
            return embeddings
            
        except Exception as e:
            print(f"⚠ 加载缓存文件时出错: {e}，将重新计算")
            return None
    
    def _precompute_embeddings(self, batch_size: int = 10):
        """
        预计算所有 clip 的 embeddings（带缓存功能）
        
        Args:
            batch_size: 批处理大小，视频模式建议使用较小的值（如 5-10），文本模式可以更大
        """
        embedding_type = "视频" if self.use_video_embedding else "文本描述"
        
        # 尝试从缓存加载
        if not self.force_recompute:
            print(f"正在检查 {embedding_type} embeddings 缓存...")
            cached_embeddings = self._load_embeddings_cache()
            
            if cached_embeddings is not None:
                self.clip_embeddings = cached_embeddings
                print(f"✓ 成功从缓存加载 embeddings，跳过重新计算")
                return
        else:
            print(f"force_recompute=True，将忽略缓存并重新计算")
        
        print(f"正在预计算 {len(self.clips_data)} 个片段的 {embedding_type} embeddings...")
        
        # 根据选项构造输入数据
        if self.use_video_embedding:
            # 使用视频文件路径
            inputs = [{"video": clip["clip_path"]} for clip in self.clips_data]
            # 视频处理较慢，使用较小的批处理大小
            if batch_size > 10:
                batch_size = 10
                print(f"视频模式下自动调整批处理大小为 {batch_size}")
        else:
            # 使用文本描述
            inputs = [{"text": clip["description"]} for clip in self.clips_data]
        
        # 分批处理
        all_embeddings = []
        total_batches = (len(inputs) + batch_size - 1) // batch_size
        
        try:
            api_url = f"{self.api_base_url}/get_embeddings"
            
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                print(f"处理批次 {batch_num}/{total_batches} ({len(batch)} 个片段)...")
                
                # 构造请求数据
                request_data = {"inputs": batch}
                
                response = requests.post(api_url, json=request_data, timeout=300)  # 5分钟超时
                response.raise_for_status()
                
                result = response.json()
                batch_embeddings = result["embeddings"]
                all_embeddings.extend(batch_embeddings)
                
                print(f"  ✓ 批次 {batch_num} 完成")
            
            # 转换为 numpy 数组以便后续计算
            self.clip_embeddings = np.array(all_embeddings)
            
            print(f"✓ 成功预计算所有 {embedding_type} embeddings，形状: {self.clip_embeddings.shape}")
            
            # 保存到缓存
            self._save_embeddings_cache(self.clip_embeddings)
            
        except requests.exceptions.RequestException as e:
            print(f"✗ API 请求错误: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"  错误详情: {error_detail}")
                except:
                    print(f"  响应内容: {e.response.text[:500]}")
            raise
        except Exception as e:
            print(f"✗ 处理响应时出错: {e}")
            raise
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        获取查询文本的 embedding
        
        Args:
            query: 用户查询文本
            
        Returns:
            query 的 embedding 向量
        """
        # 构造请求数据
        request_data = {
            "inputs": [{"text": query}]
        }
        
        try:
            api_url = f"{self.api_base_url}/get_embeddings"
            response = requests.post(api_url, json=request_data)
            response.raise_for_status()
            
            result = response.json()
            embedding = result["embeddings"][0]  # 只有一个 query，取第一个
            
            return np.array(embedding)
            
        except requests.exceptions.RequestException as e:
            print(f"API 请求错误: {e}")
            raise
        except Exception as e:
            print(f"处理响应时出错: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        根据查询文本搜索最相关的视频片段
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个最相关的片段
            
        Returns:
            包含片段信息和相似度分数的列表
        """
        embedding_type = "视频 Embedding" if self.use_video_embedding else "文本 Embedding"
        
        print(f"\n查询: {query}")
        print(f"检索模式: {embedding_type}")
        print(f"在 {len(self.clips_data)} 个片段中搜索...")
        
        # 获取查询的 embedding
        query_embedding = self._get_query_embedding(query)
        
        # 计算查询与所有片段的相似度
        # 使用矩阵乘法: query_embedding @ clip_embeddings.T
        similarity_scores = query_embedding @ self.clip_embeddings.T
        
        # 将相似度分数添加到片段信息中
        results = []
        for i, clip in enumerate(self.clips_data):
            result = clip.copy()
            result["similarity_score"] = float(similarity_scores[i])
            results.append(result)
        
        # 按相似度降序排序
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # 返回 top k 个结果
        top_results = results[:top_k]
        
        return top_results
    
    def print_search_results(self, results: List[Dict], show_full_path: bool = False):
        """
        打印搜索结果
        
        Args:
            results: 搜索结果列表
            show_full_path: 是否显示完整路径
        """
        if not results:
            print("没有找到匹配的片段")
            return
        
        print(f"\n找到 {len(results)} 个最相关的片段:")
        print("=" * 100)
        
        for i, result in enumerate(results, 1):
            clip_path = result["clip_path"] if show_full_path else result["clip_path"].split("/")[-1]
            
            print(f"\n排名 #{i}")
            print(f"  片段 ID: {result['clip_id']}")
            print(f"  相似度: {result['similarity_score']:.4f}")
            print(f"  时间范围: {result['start_time']:.2f}s - {result['end_time']:.2f}s (时长: {result['duration']:.2f}s)")
            print(f"  描述: {result['description']}")
            if show_full_path:
                print(f"  路径: {clip_path}")
            else:
                print(f"  文件名: {clip_path}")
        
        print("=" * 100)
    
    def get_clip_by_id(self, clip_id: int) -> Optional[Dict]:
        """
        根据 clip_id 获取片段信息
        
        Args:
            clip_id: 片段 ID
            
        Returns:
            片段信息字典，如果不存在则返回 None
        """
        for clip in self.clips_data:
            if clip["clip_id"] == clip_id:
                return clip.copy()
        return None
    
    def get_clips_in_time_range(self, start_time: float, end_time: float) -> List[Dict]:
        """
        获取指定时间范围内的所有片段
        
        Args:
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            片段信息列表
        """
        results = []
        for clip in self.clips_data:
            # 检查片段是否与指定时间范围有重叠
            if clip["start_time"] < end_time and clip["end_time"] > start_time:
                results.append(clip.copy())
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        if not self.clips_data:
            return {}
        
        durations = [clip["duration"] for clip in self.clips_data]
        
        stats = {
            "total_clips": len(self.clips_data),
            "source_video": self.source_video,
            "total_duration": sum(durations),
            "avg_duration": np.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "time_range": {
                "start": self.clips_data[0]["start_time"] if self.clips_data else 0,
                "end": self.clips_data[-1]["end_time"] if self.clips_data else 0
            }
        }
        
        return stats
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n数据集统计信息:")
        print("=" * 80)
        print(f"源视频: {stats.get('source_video', 'N/A')}")
        print(f"总片段数: {stats.get('total_clips', 0)}")
        print(f"总时长: {stats.get('total_duration', 0):.2f} 秒")
        print(f"平均片段时长: {stats.get('avg_duration', 0):.2f} 秒")
        print(f"最短片段: {stats.get('min_duration', 0):.2f} 秒")
        print(f"最长片段: {stats.get('max_duration', 0):.2f} 秒")
        time_range = stats.get('time_range', {})
        print(f"时间范围: {time_range.get('start', 0):.2f}s - {time_range.get('end', 0):.2f}s")
        print("=" * 80)


def main():
    """示例用法"""
    # 初始化记忆系统
    json_path = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/output_clips/chunwu_middle/chunwu_middle_clips_info.json"
    
    # 示例1: 使用文本描述的 Embedding 进行检索（默认）
    print("=" * 100)
    print("示例1: 使用文本描述 Embedding 进行检索（带缓存）")
    print("=" * 100)
    # 第一次运行会计算并缓存 embeddings，第二次运行会直接从缓存加载
    memory_text = ClipMemory(
        json_path, 
        use_video_embedding=False, 
        batch_size=50,
        cache_dir="./.cache",  # 指定缓存目录
        force_recompute=False  # False=使用缓存，True=强制重新计算
    )
    memory_text.print_statistics()
    
    query = "一个女孩在厨房做饭"
    results_text = memory_text.search(query, top_k=3)
    memory_text.print_search_results(results_text, show_full_path=False)
    
    print("\n\n")
    
    # 示例2: 使用视频 Embedding 进行检索（带缓存）
    print("=" * 100)
    print("示例2: 使用视频 Embedding 进行检索（带缓存）")
    print("=" * 100)
    # 视频模式使用较小的批处理大小，避免服务器过载
    # 第一次运行会比较慢（需要编码所有视频），第二次运行会很快（从缓存加载）
    memory_video = ClipMemory(
        json_path, 
        use_video_embedding=True, 
        batch_size=5,
        cache_dir="./.cache",  # 指定缓存目录
        force_recompute=False  # False=使用缓存，True=强制重新计算
    )
    
    results_video = memory_video.search(query, top_k=3)
    memory_video.print_search_results(results_video, show_full_path=False)


if __name__ == "__main__":
    main()

