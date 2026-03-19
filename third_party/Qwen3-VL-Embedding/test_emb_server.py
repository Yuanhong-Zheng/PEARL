"""
测试 Qwen3-VL-Embedding API 服务的示例脚本
"""
import requests
import json


def test_compute_similarity():
    """测试相似度计算 API"""
    
    url = "http://localhost:5000/compute_similarity"
    
    # 准备请求数据
    data = {
        "queries": [
            {"image": "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache/chunwu_0120/XiaoMing.jpg"},
            # {"text": "This person's name is XiaoJing.", "video": ""},
        ],
        "documents": [
            # {"image": "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache/5limi/XiaoJing.jpg"},
            {"video": "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/video-level/clips/jianshen1/jianshen1_scene_098.mp4"},
            {"video": "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/video-level/clips/jianshen1/jianshen1_scene_005.mp4"},
            {"video": "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/video-level/clips/jianshen1/jianshen1_scene_009.mp4"},
            {"video": "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/video-level/clips/jianshen1/jianshen1_scene_017.mp4"},
        ]
    }
    
    # 发送 POST 请求
    print("发送请求到:", url)
    print("请求数据:", json.dumps(data, indent=2, ensure_ascii=False))
    
    response = requests.post(url, json=data)
    
    # 打印结果
    print("\n状态码:", response.status_code)
    print("响应结果:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    
    if response.status_code == 200:
        similarity_scores = response.json()["similarity_scores"]
        print("\n相似度分数矩阵:")
        print(f"查询数量: {len(data['queries'])}")
        print(f"文档数量: {len(data['documents'])}")
        for i, scores in enumerate(similarity_scores):
            print(f"查询 {i+1} 与各文档的相似度: {scores}")





if __name__ == "__main__":
    test_compute_similarity()

