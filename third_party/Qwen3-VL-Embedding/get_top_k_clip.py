"""
测试 Qwen3-VL-Embedding API 服务的示例脚本
"""
import requests
import json
import os
from pathlib import Path


def get_all_mp4_files(folder_path):
    """从文件夹获取所有 mp4 文件"""
    mp4_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"文件夹不存在: {folder_path}")
    
    # 递归搜索所有 mp4 文件
    for mp4_file in folder.rglob("*.mp4"):
        mp4_files.append(str(mp4_file.absolute()))
    
    return sorted(mp4_files)


def test_compute_similarity_top_k(query_video, documents_folder, top_k=5, batch_size=10):
    """
    测试相似度计算 API，找出 top k 个最相关的视频
    采用批处理方式避免 OOM
    
    Args:
        query_video: 查询视频的路径
        documents_folder: 包含所有文档视频的文件夹路径
        top_k: 返回前 k 个最相关的视频
        batch_size: 每批处理的文档数量，默认 10
    """
    
    url = "http://localhost:5000/compute_similarity"
    
    # 从文件夹读取所有 mp4 文件
    print(f"正在从文件夹读取所有 mp4 文件: {documents_folder}")
    mp4_files = get_all_mp4_files(documents_folder)
    print(f"找到 {len(mp4_files)} 个 mp4 文件")
    print(f"批处理大小: 每批 {batch_size} 个文件\n")
    
    if not mp4_files:
        print("错误: 文件夹中没有找到 mp4 文件")
        return
    
    # 存储所有视频的相似度分数
    all_video_scores = []
    
    # 分批处理
    total_batches = (len(mp4_files) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(mp4_files))
        batch_files = mp4_files[start_idx:end_idx]
        
        print(f"处理批次 {batch_idx + 1}/{total_batches} (文件 {start_idx + 1}-{end_idx}/{len(mp4_files)})")
        
        # 准备当前批次的请求数据
        data = {
            "queries": [
                {"video":query_video,"instruction":"Pay attention to the gestures of people in the video",}
            ],
            "documents": [
                {"video": video_path,} for video_path in batch_files
            ]
        }
        
        try:
            # 发送 POST 请求
            response = requests.post(url, json=data, timeout=300)  # 5分钟超时
            
            if response.status_code == 200:
                result = response.json()
                similarity_scores = result["similarity_scores"][0]  # 只有一个查询，取第一行
                
                # 将当前批次的结果添加到总结果中
                for video_path, score in zip(batch_files, similarity_scores):
                    all_video_scores.append((video_path, score))
                
                print(f"  ✓ 批次 {batch_idx + 1} 完成")
            else:
                print(f"  ✗ 批次 {batch_idx + 1} 失败: HTTP {response.status_code}")
                print(f"    错误信息: {response.text}")
                
        except Exception as e:
            print(f"  ✗ 批次 {batch_idx + 1} 出错: {str(e)}")
            continue
    
    print(f"\n总共处理了 {len(all_video_scores)} 个视频")
    
    if not all_video_scores:
        print("错误: 没有成功处理任何视频")
        return
    
    # 按相似度降序排序
    all_video_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 输出 top k 结果
    print(f"\n{'='*80}")
    print(f"Top {top_k} 最相关的视频:")
    print(f"{'='*80}\n")
    
    for rank, (video_path, score) in enumerate(all_video_scores[:top_k], 1):
        print(f"排名 {rank}:")
        print(f"  文件名: {os.path.basename(video_path)}")
        print(f"  完整路径: {video_path}")
        print(f"  相似度分数: {score:.4f}")
        print()
    
    # 如果需要，也可以保存结果到文件
    save_results = False  # 设置为 True 如果需要保存结果
    if save_results:
        output_file = "similarity_results.json"
        results_data = {
            "query_video": query_video,
            "top_k": top_k,
            "total_videos": len(all_video_scores),
            "results": [
                {
                    "rank": rank,
                    "video_path": video_path,
                    "similarity_score": float(score)
                }
                for rank, (video_path, score) in enumerate(all_video_scores[:top_k], 1)
            ]
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"结果已保存到: {output_file}")



if __name__ == "__main__":
    # 配置参数
    query_video = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/Qwen3-VL-Embedding/data/shouyu/你好2-1.mp4"
    documents_folder = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/Qwen3-VL-Embedding/data/shouyu"
    top_k = 5  # 返回前 5 个最相关的视频
    batch_size = 10  # 每批处理 10 个视频文件，可根据内存情况调整（减小避免 OOM）
    
    # 运行测试
    test_compute_similarity_top_k(query_video, documents_folder, top_k, batch_size)
