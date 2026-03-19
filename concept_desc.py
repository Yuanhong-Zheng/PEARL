"""
Generate distinctive descriptions for each concept in concept_db.json.
These descriptions are used during retrieval to replace custom concept names
and help the embedding model better understand the query.
"""
import json
import argparse
from pathlib import Path
from openai import OpenAI


def generate_distinctive_description(client, model_path, image_path, concept_name, original_description):
    """
    Generate a distinctive description for a concept image.
    
    Args:
        client: OpenAI client
        model_path: Model path
        image_path: Image path
        concept_name: Concept name
        original_description: Original description
        
    Returns:
        Distinctive description text
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
    Process a concept database and generate a distinctive description for each concept.
    
    Args:
        concept_db_path: Path to concept_db.json
        api_base_url: API service URL
        model_path: Model path
        force_regenerate: Whether to forcibly regenerate existing descriptions
    """
    print("=" * 80)
    print("Generating distinctive descriptions for concepts...")
    print("=" * 80)
    
    # Resolve to an absolute path
    concept_db_path = str(Path(concept_db_path).resolve())
    
    # Load the concept database
    print(f"\nLoading concept database: {concept_db_path}")
    with open(concept_db_path, 'r', encoding='utf-8') as f:
        db_data = json.load(f)
    
    concepts = db_data.get('concepts', [])
    print(f"Found {len(concepts)} concept(s)\n")
    
    if not concepts:
        print("⚠ The concept database is empty; nothing to process")
        return
    
    # Initialize the OpenAI client
    print("Initializing model API...")
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base_url,
        timeout=3600
    )
    print("✓ Model API initialized")
    print(f"  API URL: {api_base_url}")
    print(f"  Model path: {model_path}\n")
    
    # Statistics
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    # Generate a distinctive description for each concept
    for i, concept in enumerate(concepts, 1):
        concept_name = concept.get('concept_name', 'Unknown')
        frame_path = concept.get('frame_path')
        original_description = concept.get('description', '')
        
        print(f"[{i}/{len(concepts)}] Processing concept: {concept_name}")
        
        if not frame_path:
            print("  ⚠ No frame_path found, skipping\n")
            skipped_count += 1
            continue
        
        # Convert relative paths to absolute paths
        frame_path_obj = Path(frame_path)
        if not frame_path_obj.is_absolute():
            # Resolve relative to the directory containing concept_db.json
            db_dir = Path(concept_db_path).parent
            frame_path = str((db_dir / frame_path).resolve())
        
        print(f"  Image path: {frame_path}")
        
        # Verify the image exists
        if not Path(frame_path).exists():
            print("  ✗ Image file does not exist, skipping\n")
            skipped_count += 1
            continue
        
        # Skip concepts that already have a description unless regeneration is forced
        if not force_regenerate and 'retrieval_description' in concept and concept['retrieval_description']:
            print("  ✓ Distinctive description already exists:")
            print(f"    {concept['retrieval_description']}")
            print("  Skipping (use --skip-existing to keep existing descriptions)\n")
            skipped_count += 1
            continue
        
        try:
            # Generate the description
            print("  Generating distinctive description...")
            description = generate_distinctive_description(
                client=client,
                model_path=model_path,
                image_path=frame_path,
                concept_name=concept_name,
                original_description=original_description
            )
            
            # Store it in the concept entry
            concept['retrieval_description'] = description
            
            print("  ✓ Generated successfully:")
            print(f"    {description}\n")
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Generation failed: {e}\n")
            concept['retrieval_description'] = ""
            failed_count += 1
    
    # Save the updated database
    if processed_count > 0:
        print("\nSaving updated concept database...")
        with open(concept_db_path, 'w', encoding='utf-8') as f:
            json.dump(db_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved to: {concept_db_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Done! Statistics:")
    print(f"  Total concepts: {len(concepts)}")
    print(f"  Generated successfully: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Failed: {failed_count}")
    print("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate distinctive descriptions for a concept database so abstract concept names can be replaced during retrieval"
    )
    parser.add_argument(
        "--concept_db_path",
        type=str,
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache/5limi/concept_db.json",
        help="Path to concept_db.json (default: ./.cache/concept_db.json)"
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default="http://127.0.0.1:22003/v1",
        help="API service URL (default: http://127.0.0.1:22003/v1)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/models/OpenGVLab/InternVL3_5-8B",
        help="Model path"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing descriptions instead of regenerating them (default: False, meaning regenerate all descriptions)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    process_concept_database(
        concept_db_path=args.concept_db_path,
        api_base_url=args.api_base_url,
        model_path=args.model_path,
        force_regenerate=not args.skip_existing  # Regenerate by default unless --skip-existing is provided
    )


if __name__ == "__main__":
    main()
