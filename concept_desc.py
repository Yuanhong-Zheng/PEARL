"""
为 concept_db.json 中的每个概念生成特征性描述
用于在检索时替换自定义概念名称，帮助 embedding 模型更好地理解查询内容
"""
import json
import argparse
from pathlib import Path
from openai import OpenAI


def generate_distinctive_description(client, model_path, image_path, concept_name, original_description):
    """
    为概念图像生成特征性描述
    
    Args:
        client: OpenAI client
        model_path: 模型路径
        image_path: 图像路径
        concept_name: 概念名称
        original_description: 原始描述
        
    Returns:
        特征性描述文本
    """
    prompt = f"""Based on the image and the original description provided, generate a concise visual description of this character/object that focuses on PERMANENT/STABLE features for video clip retrieval.

Original description: "{original_description}"
Concept name: {concept_name}

Your task:
1. Use the original description to understand WHICH character/object to focus on in the image
2. Generate a description focusing on STABLE features that DON'T change throughout the video:
   - Gender (male/female/other)
   - Face features (eye shape, facial structure, distinctive marks)
   - Hair (color, length, style if distinctive)
   - Body type (build)
   - Age appearance (young/middle-aged/elderly)

AVOID or minimize:
- Clothing details (they change in long videos)
- Accessories (they may be removed)
- Temporary expressions or poses
- Background, location, surroundings, or nearby objects in the scene
- Relative position or size compared to objects/environment in the scene

Requirements:
- Keep it concise and simple (1 sentence, around 10-15 words)
- Focus on features that remain consistent across different scenes
- Write in English using simple descriptive terms
- Use third person (e.g., "a young male with...", "the girl with...")
- Make it natural enough to replace the concept name in a question

Please provide the distinctive visual description focusing on PERMANENT features:"""
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"file://{image_path}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model=model_path,
        messages=messages,
        max_tokens=512,
        temperature=0.7
    )
    
    description = response.choices[0].message.content.strip()
    return description


def process_concept_database(
    concept_db_path: str,
    api_base_url: str = "http://127.0.0.1:22003/v1",
    model_path: str = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/OpenGVLab/InternVL3_5-8B",
    force_regenerate: bool = False
):
    """
    处理概念数据库，为每个概念生成特征性描述
    
    Args:
        concept_db_path: concept_db.json 文件路径
        api_base_url: API 服务地址
        model_path: 模型路径
        force_regenerate: 是否强制重新生成已有的描述
    """
    print("=" * 80)
    print("开始为概念生成特征性描述...")
    print("=" * 80)
    
    # 转换为绝对路径
    concept_db_path = str(Path(concept_db_path).resolve())
    
    # 读取概念数据库
    print(f"\n读取概念数据库: {concept_db_path}")
    with open(concept_db_path, 'r', encoding='utf-8') as f:
        db_data = json.load(f)
    
    concepts = db_data.get('concepts', [])
    print(f"找到 {len(concepts)} 个概念\n")
    
    if not concepts:
        print("⚠ 概念数据库为空，无需处理")
        return
    
    # 初始化 OpenAI Client
    print("初始化模型 API...")
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base_url,
        timeout=3600
    )
    print(f"✓ 模型 API 初始化完成")
    print(f"  API 地址: {api_base_url}")
    print(f"  模型路径: {model_path}\n")
    
    # 统计信息
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # 为每个概念生成特征性描述
    for i, concept in enumerate(concepts, 1):
        concept_name = concept.get('concept_name', 'Unknown')
        frame_path = concept.get('frame_path')
        original_description = concept.get('description', '')
        
        print(f"[{i}/{len(concepts)}] 处理概念: {concept_name}")
        
        if not frame_path:
            print(f"  ⚠ 没有 frame_path，跳过\n")
            skipped_count += 1
            continue
        
        # 将相对路径转换为绝对路径
        frame_path_obj = Path(frame_path)
        if not frame_path_obj.is_absolute():
            # 相对于 concept_db.json 所在目录
            db_dir = Path(concept_db_path).parent
            frame_path = str((db_dir / frame_path).resolve())
        
        print(f"  图像路径: {frame_path}")
        
        # 检查图像是否存在
        if not Path(frame_path).exists():
            print(f"  ✗ 图像文件不存在，跳过\n")
            skipped_count += 1
            continue
        
        # 如果已经有特征性描述且不强制重新生成，跳过
        if not force_regenerate and 'retrieval_description' in concept and concept['retrieval_description']:
            print(f"  ✓ 已存在特征性描述:")
            print(f"    {concept['retrieval_description']}")
            print(f"  跳过（默认会重新生成，使用 --skip-existing 跳过已有描述）\n")
            skipped_count += 1
            continue
        
        try:
            # 生成特征性描述
            print(f"  正在生成特征性描述...")
            description = generate_distinctive_description(
                client=client,
                model_path=model_path,
                image_path=frame_path,
                concept_name=concept_name,
                original_description=original_description
            )
            
            # 添加到概念中
            concept['retrieval_description'] = description
            
            print(f"  ✓ 生成成功:")
            print(f"    {description}\n")
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ 生成失败: {e}\n")
            concept['retrieval_description'] = ""
            failed_count += 1
    
    # 保存更新后的数据库
    if processed_count > 0:
        print(f"\n保存更新后的概念数据库...")
        with open(concept_db_path, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 已保存到: {concept_db_path}")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("处理完成！统计信息:")
    print(f"  总概念数: {len(concepts)}")
    print(f"  成功生成: {processed_count}")
    print(f"  跳过: {skipped_count}")
    print(f"  失败: {failed_count}")
    print("=" * 80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="为概念数据库生成特征性描述，用于检索时替换抽象的概念名称"
    )
    parser.add_argument(
        "--concept_db_path",
        type=str,
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache/5limi/concept_db.json",
        help="concept_db.json 文件路径（默认: ./.cache/concept_db.json）"
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default="http://127.0.0.1:22003/v1",
        help="API 服务地址（默认: http://127.0.0.1:22003/v1）"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/OpenGVLab/InternVL3_5-8B",
        help="模型路径"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="跳过已有描述，不重新生成（默认: False，即默认重新生成所有描述）"
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    process_concept_database(
        concept_db_path=args.concept_db_path,
        api_base_url=args.api_base_url,
        model_path=args.model_path,
        force_regenerate=not args.skip_existing  # 默认重新生成，除非指定 --skip-existing
    )


if __name__ == "__main__":
    main()

