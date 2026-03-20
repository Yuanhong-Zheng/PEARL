import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict
import subprocess
from utils import extract_video_frame


class ConceptDatabase:
    """
    A simple concept database for storing and querying concepts defined in videos.

    Each concept includes:
    - concept_name: concept name
    - description: short model-generated description
    - frame_path: stored path to the extracted frame image
    - timestamp: timestamp
    - video_path: video path

    Features:
    - Automatically extracts frames from videos at specified timestamps
    - Saves frame images using concept_name as the filename
    - Supports insertion and query operations
    """
    
    def __init__(self, db_path: str = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache/concept_db.json", frame_dir: str = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache"):
        """
        Initialize the database.
        
        Args:
            db_path: Path to the JSON database file
            frame_dir: Directory used to store extracted concept frames
        """
        self.db_path = Path(db_path)
        self.frame_dir = Path(frame_dir)
        
        # Create the frame storage directory
        self.frame_dir.mkdir(exist_ok=True)
        
        # Load or initialize the database
        self.data = self._load_db()
    
    def _load_db(self) -> Dict:
        """Load the database file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: database file {self.db_path} is malformed; creating a new database")
                return {"concepts": []}
        else:
            return {"concepts": []}
    
    def _save_db(self):
        """Save the database to disk."""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def _extract_frame(self, video_path: str, timestamp: str, output_path: str) -> bool:
        """
        Extract a frame from the video at the specified timestamp.
        
        Args:
            video_path: Video file path
            timestamp: Timestamp in HH:MM:SS format
            output_path: Output image path
            
        Returns:
            bool: True if extraction succeeded, otherwise False
        """
        return extract_video_frame(
            source_video=video_path,
            timestamp=timestamp,
            output_path=output_path,
            verbose=True
        )

    def _safe_concept_filename(self, concept_name: str) -> str:
        """Convert a concept name to a filesystem-safe base filename."""
        return concept_name.replace(" ", "_")

    def _extract_clip(self, video_path: str, start_time: str, end_time: str, output_path: str) -> bool:
        """
        Extract a video clip from the specified time range.
        
        Args:
            video_path: Video file path
            start_time: Start timestamp in HH:MM:SS format
            end_time: End timestamp in HH:MM:SS format
            output_path: Output video path
            
        Returns:
            bool: True if extraction succeeded, otherwise False
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', start_time,
                '-to', end_time,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',  # Overwrite existing files
                output_path
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True
            )
            
            if os.path.exists(output_path):
                print(f"✓ Successfully extracted clip: {output_path}")
                return True
            else:
                print("✗ Failed to extract clip: output file does not exist")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"✗ ffmpeg failed to extract clip: {e}")
            print(f"stderr: {e.stderr.decode()}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error while extracting clip: {e}")
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
        Add a new concept to the database.

        Supports two modes:
        - Frame mode: provide timestamp to extract a single frame
        - Clip mode: provide start_time + end_time to extract a video clip
        
        Args:
            concept_name: Concept name
            description: Model-generated description
            video_path: Video path
            timestamp: Timestamp in HH:MM:SS format for frame mode
            start_time: Start timestamp in HH:MM:SS format for clip mode
            end_time: End timestamp in HH:MM:SS format for clip mode
            additional_info: Optional extra information
            
        Returns:
            bool: True if the concept was added successfully, otherwise False
        """
        try:
            if start_time and end_time:
                # Clip mode: extract a video clip
                clip_filename = f"{self._safe_concept_filename(concept_name)}.mp4"
                target_path = self.frame_dir / clip_filename
                
                if not self._extract_clip(video_path, start_time, end_time, str(target_path)):
                    print(f"✗ Unable to extract clip from video; skipping concept {concept_name}")
                    return False
                
                # Create a concept record (concept_type = "clip")
                concept_entry = {
                    "concept_name": concept_name,
                    "description": description,
                    "frame_path": str(target_path),  # Preserve backward compatibility by storing the clip path here
                    "concept_type": "clip",
                    "start_time": start_time,
                    "end_time": end_time,
                    "video_path": video_path,
                }
            else:
                # Frame mode: extract a single frame
                frame_filename = f"{self._safe_concept_filename(concept_name)}.jpg"
                target_frame_path = self.frame_dir / frame_filename
                
                if not self._extract_frame(video_path, timestamp, str(target_frame_path)):
                    print(f"✗ Unable to extract frame from video; skipping concept {concept_name}")
                    return False
                
                # Create a concept record (concept_type = "frame")
                concept_entry = {
                    "concept_name": concept_name,
                    "description": description,
                    "frame_path": str(target_frame_path),
                    "concept_type": "frame",
                    "timestamp": timestamp,
                    "video_path": video_path,
                }
            
            # Add extra information
            if additional_info:
                concept_entry.update(additional_info)
            
            # Append to the database
            self.data["concepts"].append(concept_entry)
            
            # Save to disk
            self._save_db()
            
            print(f"✓ Successfully added concept: {concept_name} (type: {concept_entry['concept_type']})")
            return True
            
        except Exception as e:
            print(f"✗ Failed to add concept {concept_name}: {e}")
            return False
    
    def query_by_name(self, concept_name: str) -> Optional[Dict]:
        """
        Query a concept by name.
        
        Args:
            concept_name: Concept name
            
        Returns:
            Dict: Concept information, or None if not found
        """
        for concept in self.data["concepts"]:
            if concept["concept_name"] == concept_name:
                return concept
        return None
    
    def clear_database(self):
        """Clear all concepts from the database."""
        self.data = {"concepts": []}
        self._save_db()
        print("✓ Database cleared")
    
    def extract_concept_names(self, text: str) -> List[str]:
        """
        Extract all concept names enclosed in {} from a text string.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Extracted concept names
        """
        # Match the {concept_name} pattern with a regular expression
        pattern = r'\{([^}]+)\}'
        concepts = re.findall(pattern, text)
        return concepts
    
    def add_concepts_from_annotation_file(
        self, 
        annotation_file: str,
        clear_before_add: bool = True
    ) -> int:
        """
        Add concepts in batch from an annotation file.

        This method:
        1. Reads the annotation file
        2. Iterates over all entries of type "concept definition"
        3. Extracts concept names and adds them to the database
        4. Skips duplicate concepts
        
        Args:
            annotation_file: Path to the annotation file
            clear_before_add: Whether to clear the database before insertion
            
        Returns:
            int: Number of concepts added successfully
        """
        # 1. Clear the database if requested
        if clear_before_add:
            print("\nClearing database...")
            self.clear_database()
        
        annotation_file_path = Path(annotation_file).resolve()
        project_root = Path(__file__).resolve().parent

        # 2. Load the annotation file
        print(f"\nLoading annotation file: {annotation_file}")
        with open(annotation_file_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print("✓ Annotation file loaded successfully")
        
        # 3. Extract and add concept definitions
        print("\nExtracting and adding concept definitions...")
        concept_count = 0
        skipped_count = 0
        error_count = 0
        
        for video_item in annotations:
            video_path_raw = video_item.get('video_path')
            if not video_path_raw:
                raise ValueError("Found an annotation entry with a missing video_path")

            video_path_obj = Path(video_path_raw)
            if not video_path_obj.is_absolute():
                video_path_obj = (project_root / video_path_obj).resolve()
            else:
                video_path_obj = video_path_obj.resolve()

            if not video_path_obj.exists():
                raise FileNotFoundError(f"Video file does not exist: {video_path_obj}")

            video_path = str(video_path_obj)
            timestamps = video_item.get('timestamps', [])
            
            print(f"\nProcessing video: {video_path}")
            
            for qa_item in timestamps:
                # Only process entries of type "concept definition"
                if qa_item.get('qa_type') != "concept definition":
                    continue
                
                question = qa_item.get('question', '')
                time = qa_item.get('time', '')
                start_time = qa_item.get('start_time', '')
                end_time = qa_item.get('end_time', '')
                
                # Extract concept names
                concept_names = self.extract_concept_names(question)
                
                if not concept_names:
                    print(f"  ⚠ Warning: no concept names found in question: {question}")
                    continue
                
                # Decide the mode: prefer start_time/end_time (clip mode), otherwise use time (frame mode)
                use_clip_mode = bool(start_time and end_time)
                if use_clip_mode:
                    print(f"  [Clip mode] Time range: {start_time} -> {end_time}")
                else:
                    print(f"  [Frame mode] Timestamp: {time}")
                
                # Add each concept
                for concept_name in concept_names:
                    # Check whether the concept already exists
                    existing_concept = self.query_by_name(concept_name)
                    if existing_concept:
                        print(f"  ⊗ Skipping duplicate concept: {concept_name} (already exists)")
                        skipped_count += 1
                        continue
                    
                    print(f"\n  Adding concept: {concept_name}")
                    print(f"    Question: {question[:80]}...")
                    
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
        
        # 4. Show summary statistics
        print("\n" + "-" * 80)
        print("✓ Processing complete!")
        print(f"  Added successfully: {concept_count} concept(s)")
        print(f"  Duplicates skipped: {skipped_count} concept(s)")
        print(f"  Failed to add: {error_count} concept(s)")
        assert error_count==0, "Errors occurred while adding concepts; please check the logs"
        print("-" * 80)
        
        return concept_count


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Starting concept-definition processing")
    print("=" * 80)
    
    # Configure paths
    base_cache_dir = Path("/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/.cache")
    annotation_file = Path("/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/data/annotations/apartment5.json")
    
    # Check whether the file exists
    if not annotation_file.exists():
        print(f"✗ File does not exist: {annotation_file}")
        exit(1)
    
    annotation_name = annotation_file.stem  # Filename without extension
    
    print(f"\nProcessing annotation: {annotation_file.name}")
    print("=" * 80)
    
    # Create a dedicated subdirectory for this annotation
    annotation_cache_dir = base_cache_dir / annotation_name
    annotation_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the database path and frame storage directory
    db_path = annotation_cache_dir / "concept_db.json"
    frame_dir = annotation_cache_dir
    
    print(f"\nDatabase path: {db_path}")
    print(f"Frame storage path: {frame_dir}")
    
    # Create the database instance
    db = ConceptDatabase(db_path=str(db_path), frame_dir=str(frame_dir))
    
    # Batch-add concepts from the annotation file
    concept_count = db.add_concepts_from_annotation_file(
        annotation_file=str(annotation_file),
        clear_before_add=True
    )
    
    # Show all concepts stored for the current annotation
    if concept_count > 0:
        print(f"\nAll concepts in the {annotation_name} database:")
        print("-" * 80)
        for i, concept in enumerate(db.data['concepts'], 1):
            print(f"{i}. {concept['concept_name']}")
            print(f"   Timestamp: {concept['timestamp']}")
            print(f"   Frame path: {concept['frame_path']}")
            print()
    
    print("\n" + "=" * 80)
    print(f"✓ Done! Added {concept_count} concept(s) in total")
    print("=" * 80)
