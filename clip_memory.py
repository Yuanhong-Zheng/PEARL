"""
Memory System - Video clip memory system.
Uses the Qwen3-VL-Embedding API for semantic similarity retrieval.
"""
import json
import requests
from typing import List, Dict, Optional
import numpy as np
from pathlib import Path
import hashlib


class ClipMemory:
    """Video clip memory."""
    
    def __init__(self, json_path: str, api_base_url: str = "http://localhost:5000", 
                 use_video_embedding: bool = False, batch_size: int = 10,
                 cache_dir: str = None, force_recompute: bool = False):
        """
        Initialize the memory system.
        
        Args:
            json_path: Path to the clips-info JSON file
            api_base_url: Base URL for the embedding API
            use_video_embedding: Whether to use video embeddings for retrieval
            batch_size: Batch size for precomputation
            cache_dir: Cache directory; if None, use the JSON file's directory
            force_recompute: Whether to recompute embeddings and ignore the cache
        """
        self.json_path = json_path
        self.api_base_url = api_base_url
        self.use_video_embedding = use_video_embedding
        self.clips_data = []
        self.source_video = ""
        self.clip_embeddings = None  # Stores precomputed embeddings
        self.force_recompute = force_recompute
        
        # Set the cache directory
        if cache_dir is None:
            # Default to the directory containing json_path
            cache_dir = str(Path(json_path).parent)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load clip metadata
        self._load_clips()
        
        # Precompute embeddings for all clips with caching
        self._precompute_embeddings(batch_size=batch_size)
        
    def _load_clips(self):
        """Load clip metadata from a JSON file."""
        print(f"Loading clip data: {self.json_path}")
        
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
        
        # Extract the fields we need
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
        
        print(f"Loaded {len(self.clips_data)} video clip(s)")
    
    def _get_cache_path(self) -> Path:
        """
        Generate the cache file path.
        The cache file name depends on json_path and embedding type.
        
        Returns:
            Path to the cache file
        """
        # Get the JSON filename without extension
        json_filename = Path(self.json_path).stem
        
        # Add a suffix based on the embedding type
        embedding_suffix = "video" if self.use_video_embedding else "text"
        
        # Build the cache filename
        cache_filename = f"{json_filename}_embeddings_{embedding_suffix}.npy"
        
        return self.cache_dir / cache_filename
    
    def _get_clips_hash(self) -> str:
        """
        Compute a hash of the clip data to validate the cache.
        
        Returns:
            MD5 hash of the clip data
        """
        # Extract key fields (clip_id, clip_path, description)
        clips_key_info = []
        for clip in self.clips_data:
            if self.use_video_embedding:
                # Video mode: use clip_path
                key = f"{clip['clip_id']}:{clip['clip_path']}"
            else:
                # Text mode: use description
                key = f"{clip['clip_id']}:{clip['description']}"
            clips_key_info.append(key)
        
        # Compute the hash
        clips_str = "||".join(clips_key_info)
        hash_obj = hashlib.md5(clips_str.encode('utf-8'))
        return hash_obj.hexdigest()
    
    def _save_embeddings_cache(self, embeddings: np.ndarray):
        """
        Save embeddings to the cache file.
        
        Args:
            embeddings: Embedding array to save
        """
        cache_path = self._get_cache_path()
        clips_hash = self._get_clips_hash()
        
        # Save embeddings and metadata
        cache_data = {
            'embeddings': embeddings,
            'clips_hash': clips_hash,
            'num_clips': len(self.clips_data),
            'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])
        }
        
        np.save(cache_path, cache_data, allow_pickle=True)
        print(f"✓ Embeddings cached at: {cache_path}")
    
    def _load_embeddings_cache(self) -> Optional[np.ndarray]:
        """
        Load embeddings from the cache file.
        
        Returns:
            Loaded embedding array, or None if the cache is invalid
        """
        cache_path = self._get_cache_path()
        
        # Check whether the cache file exists
        if not cache_path.exists():
            print(f"Cache file not found: {cache_path}")
            return None
        
        try:
            # Load cached data
            cache_data = np.load(cache_path, allow_pickle=True).item()
            
            # Validate cache contents
            cached_hash = cache_data.get('clips_hash', '')
            current_hash = self._get_clips_hash()
            
            if cached_hash != current_hash:
                print("⚠ Cache is stale because clip data changed; recomputing")
                return None
            
            cached_num_clips = cache_data.get('num_clips', 0)
            if cached_num_clips != len(self.clips_data):
                print(f"⚠ Cache clip count mismatch (cache: {cached_num_clips}, current: {len(self.clips_data)}); recomputing")
                return None
            
            embeddings = cache_data['embeddings']
            print(f"✓ Loaded embeddings from cache: {cache_path}")
            print(f"  Shape: {embeddings.shape}, number of clips: {len(self.clips_data)}")
            
            return embeddings
            
        except Exception as e:
            print(f"⚠ Error loading cache file: {e}; recomputing")
            return None
    
    def _precompute_embeddings(self, batch_size: int = 10):
        """
        Precompute embeddings for all clips with caching.
        
        Args:
            batch_size: Batch size; video mode typically benefits from a smaller value
        """
        embedding_type = "video" if self.use_video_embedding else "text description"
        
        # Try to load from cache first
        if not self.force_recompute:
            print(f"Checking cache for {embedding_type} embeddings...")
            cached_embeddings = self._load_embeddings_cache()
            
            if cached_embeddings is not None:
                self.clip_embeddings = cached_embeddings
                print("✓ Embeddings loaded from cache; skipping recomputation")
                return
        else:
            print("force_recompute=True, ignoring cache and recomputing")
        
        print(f"Precomputing {embedding_type} embeddings for {len(self.clips_data)} clip(s)...")
        
        # Build input payloads based on the selected mode
        if self.use_video_embedding:
            # Use video file paths
            inputs = [{"video": clip["clip_path"]} for clip in self.clips_data]
            # Video processing is slower, so cap the batch size
            if batch_size > 10:
                batch_size = 10
                print(f"Automatically reduced batch size to {batch_size} for video mode")
        else:
            # Use text descriptions
            inputs = [{"text": clip["description"]} for clip in self.clips_data]
        
        # Process in batches
        all_embeddings = []
        total_batches = (len(inputs) + batch_size - 1) // batch_size
        
        try:
            api_url = f"{self.api_base_url}/get_embeddings"
            
            for i in range(0, len(inputs), batch_size):
                batch = inputs[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} clip(s))...")
                
                # Build the request payload
                request_data = {"inputs": batch}
                
                response = requests.post(api_url, json=request_data, timeout=300)  # 5-minute timeout
                response.raise_for_status()
                
                result = response.json()
                batch_embeddings = result["embeddings"]
                all_embeddings.extend(batch_embeddings)
                
                print(f"  ✓ Batch {batch_num} finished")
            
            # Convert to a NumPy array for downstream computation
            self.clip_embeddings = np.array(all_embeddings)
            
            print(f"✓ Successfully precomputed all {embedding_type} embeddings, shape: {self.clip_embeddings.shape}")
            
            # Save to cache
            self._save_embeddings_cache(self.clip_embeddings)
            
        except requests.exceptions.RequestException as e:
            print(f"✗ API request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"  Error details: {error_detail}")
                except:
                    print(f"  Response body: {e.response.text[:500]}")
            raise
        except Exception as e:
            print(f"✗ Error while processing the response: {e}")
            raise
    
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """
        Get the embedding for a query string.
        
        Args:
            query: User query text
            
        Returns:
            Query embedding vector
        """
        # Build the request payload
        request_data = {
            "inputs": [{"text": query}]
        }
        
        try:
            api_url = f"{self.api_base_url}/get_embeddings"
            response = requests.post(api_url, json=request_data)
            response.raise_for_status()
            
            result = response.json()
            embedding = result["embeddings"][0]  # Only one query, so use the first entry
            
            return np.array(embedding)
            
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            raise
        except Exception as e:
            print(f"Error while processing the response: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for the most relevant video clips given a text query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            A list containing clip information and similarity scores
        """
        embedding_type = "video embedding" if self.use_video_embedding else "text embedding"
        
        print(f"\nQuery: {query}")
        print(f"Retrieval mode: {embedding_type}")
        print(f"Searching across {len(self.clips_data)} clip(s)...")
        
        # Get the query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Compute similarity scores between the query and all clips
        # Matrix multiplication: query_embedding @ clip_embeddings.T
        similarity_scores = query_embedding @ self.clip_embeddings.T
        
        # Attach similarity scores to clip metadata
        results = []
        for i, clip in enumerate(self.clips_data):
            result = clip.copy()
            result["similarity_score"] = float(similarity_scores[i])
            results.append(result)
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Return the top-k results
        top_results = results[:top_k]
        
        return top_results
    
    def print_search_results(self, results: List[Dict], show_full_path: bool = False):
        """
        Print search results.
        
        Args:
            results: Search results
            show_full_path: Whether to show full paths
        """
        if not results:
            print("No matching clips found")
            return
        
        print(f"\nFound {len(results)} most relevant clip(s):")
        print("=" * 100)
        
        for i, result in enumerate(results, 1):
            clip_path = result["clip_path"] if show_full_path else result["clip_path"].split("/")[-1]
            
            print(f"\nRank #{i}")
            print(f"  Clip ID: {result['clip_id']}")
            print(f"  Similarity: {result['similarity_score']:.4f}")
            print(f"  Time range: {result['start_time']:.2f}s - {result['end_time']:.2f}s (duration: {result['duration']:.2f}s)")
            print(f"  Description: {result['description']}")
            if show_full_path:
                print(f"  Path: {clip_path}")
            else:
                print(f"  Filename: {clip_path}")
        
        print("=" * 100)
    
    def get_clip_by_id(self, clip_id: int) -> Optional[Dict]:
        """
        Get clip metadata by clip_id.
        
        Args:
            clip_id: Clip ID
            
        Returns:
            Clip metadata dict, or None if not found
        """
        for clip in self.clips_data:
            if clip["clip_id"] == clip_id:
                return clip.copy()
        return None
    
    def get_clips_in_time_range(self, start_time: float, end_time: float) -> List[Dict]:
        """
        Get all clips within a specified time range.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of clip metadata
        """
        results = []
        for clip in self.clips_data:
            # Check whether the clip overlaps with the requested time range
            if clip["start_time"] < end_time and clip["end_time"] > start_time:
                results.append(clip.copy())
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Statistics dictionary
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
        """Print dataset statistics."""
        stats = self.get_statistics()
        
        print("\nDataset statistics:")
        print("=" * 80)
        print(f"Source video: {stats.get('source_video', 'N/A')}")
        print(f"Total clips: {stats.get('total_clips', 0)}")
        print(f"Total duration: {stats.get('total_duration', 0):.2f} seconds")
        print(f"Average clip duration: {stats.get('avg_duration', 0):.2f} seconds")
        print(f"Shortest clip: {stats.get('min_duration', 0):.2f} seconds")
        print(f"Longest clip: {stats.get('max_duration', 0):.2f} seconds")
        time_range = stats.get('time_range', {})
        print(f"Time range: {time_range.get('start', 0):.2f}s - {time_range.get('end', 0):.2f}s")
        print("=" * 80)


def main():
    """Example usage."""
    # Initialize the memory system
    json_path = "/mnt/shared-storage-user/mineru2-shared/zqt/zqt2/PSVBench/0121/output_clips/chunwu_middle/chunwu_middle_clips_info.json"
    
    # Example 1: retrieve using text-description embeddings (default)
    print("=" * 100)
    print("Example 1: Retrieval with text-description embeddings (with caching)")
    print("=" * 100)
    # The first run computes and caches embeddings; later runs load from cache
    memory_text = ClipMemory(
        json_path, 
        use_video_embedding=False, 
        batch_size=50,
        cache_dir="./.cache",  # Explicit cache directory
        force_recompute=False  # False = use cache, True = force recomputation
    )
    memory_text.print_statistics()
    
    query = "A girl is cooking in the kitchen"
    results_text = memory_text.search(query, top_k=3)
    memory_text.print_search_results(results_text, show_full_path=False)
    
    print("\n\n")
    
    # Example 2: retrieve using video embeddings (with caching)
    print("=" * 100)
    print("Example 2: Retrieval with video embeddings (with caching)")
    print("=" * 100)
    # Video mode uses a smaller batch size to avoid overloading the server
    # The first run is slower because all videos must be encoded; later runs use the cache
    memory_video = ClipMemory(
        json_path, 
        use_video_embedding=True, 
        batch_size=5,
        cache_dir="./.cache",  # Explicit cache directory
        force_recompute=False  # False = use cache, True = force recomputation
    )
    
    results_video = memory_video.search(query, top_k=3)
    memory_video.print_search_results(results_video, show_full_path=False)


if __name__ == "__main__":
    main()
