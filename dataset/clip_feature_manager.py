import torch
import clip
import os
import pickle
import hashlib
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)

class CLIPFeatureManager:
    """
    Manages CLIP feature extraction and caching for efficient training.
    Caches CLIP embeddings to avoid repeated computation during training.
    """
    
    def __init__(
        self, 
        cache_dir: str, 
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_cache_size_mb: int = 1000
    ):
        """
        Initialize CLIP Feature Manager.
        
        Args:
            cache_dir: Directory to store cached CLIP features
            clip_model_name: Name of CLIP model to use
            device: Device to run CLIP model on
            max_cache_size_mb: Maximum cache size in MB before LRU eviction
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.clip_model_name = clip_model_name
        self.device = device
        self.max_cache_size_mb = max_cache_size_mb
        
        # Load CLIP model
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Get embedding dimension
        self.embed_dim = self._get_clip_embed_dim()
        
        # In-memory cache with LRU eviction
        self.memory_cache = OrderedDict()
        self.cache_size_bytes = 0
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        
        # Cache file naming convention
        self.cache_file_prefix = f"clip_features_{clip_model_name.replace('/', '_')}"
        
        logger.info(f"Initialized CLIPFeatureManager with model {clip_model_name}, "
                   f"embed_dim={self.embed_dim}, cache_dir={cache_dir}")
    
    def _get_clip_embed_dim(self) -> int:
        """Get CLIP embedding dimension based on model."""
        if hasattr(self.clip_model, 'text_projection'):
            return self.clip_model.text_projection.shape[0]
        
        # Fallback dimensions
        if "ViT-B" in self.clip_model_name:
            return 512
        elif "ViT-L" in self.clip_model_name:
            return 768
        else:
            return 512
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.clip_model_name}:{text}".encode()).hexdigest()
    
    def _get_cache_filename(self, dataset_type: str, subset_id: str) -> Optional[Path]:
        """Get cache filename for a dataset subset."""
        if self.cache_dir is None:
            return None
        filename = f"{self.cache_file_prefix}_{dataset_type}_{subset_id}.pkl"
        return self.cache_dir / filename
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts using CLIP, with caching.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            torch.Tensor: CLIP embeddings of shape [len(texts), embed_dim]
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check memory cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.memory_cache:
                # Move to end (LRU)
                self.memory_cache.move_to_end(cache_key)
                embeddings.append(self.memory_cache[cache_key])
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts
        if uncached_texts:
            with torch.no_grad():
                tokens = clip.tokenize(uncached_texts, truncate=True).to(self.device)
                new_embeddings = self.clip_model.encode_text(tokens).cpu().float()
            
            # Update embeddings and cache
            for idx, text_idx in enumerate(uncached_indices):
                embedding = new_embeddings[idx]
                embeddings[text_idx] = embedding
                
                # Add to memory cache
                cache_key = self._get_cache_key(uncached_texts[idx])
                self._add_to_memory_cache(cache_key, embedding)
        
        # Stack all embeddings
        return torch.stack(embeddings)
    
    def _add_to_memory_cache(self, key: str, embedding: torch.Tensor):
        """Add embedding to memory cache with LRU eviction."""
        embedding_size = embedding.element_size() * embedding.nelement()
        
        # Evict oldest entries if needed
        while self.cache_size_bytes + embedding_size > self.max_cache_size_bytes and self.memory_cache:
            oldest_key, oldest_embedding = self.memory_cache.popitem(last=False)
            old_size = oldest_embedding.element_size() * oldest_embedding.nelement()
            self.cache_size_bytes -= old_size
        
        # Add new entry
        self.memory_cache[key] = embedding
        self.cache_size_bytes += embedding_size
    
    def precompute_dataset_features(
        self, 
        dataset_type: str,
        text_data: Dict[str, List[str]],
        subset_id: str
    ) -> Dict[str, torch.Tensor]:
        """
        Precompute and cache CLIP features for an entire dataset subset.
        
        Args:
            dataset_type: Type of dataset (e.g., "adt")
            text_data: Dictionary mapping categories to lists of texts
                      e.g., {"object_category": [...], "scene_bbox_categories": [...]}
            subset_id: Identifier for this subset (e.g., "P01_mug_train")
            
        Returns:
            Dictionary mapping categories to feature tensors
        """
        cache_file = self._get_cache_filename(dataset_type, subset_id)

        # Try to load from disk
        if cache_file is not None and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded CLIP features from {cache_file}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        # Compute features
        logger.info(f"Computing CLIP features for {dataset_type} subset {subset_id}")
        feature_dict = {}
        
        for category, texts in text_data.items():
            if texts:
                # Remove duplicates while preserving order
                unique_texts = list(dict.fromkeys(texts))
                logger.info(f"Encoding {len(unique_texts)} unique texts for {category}")
                
                # Encode all unique texts
                features = self.encode_texts(unique_texts)
                
                # Create mapping from text to feature
                text_to_feature = {text: features[i] for i, text in enumerate(unique_texts)}
                
                # Map back to original order
                feature_list = [text_to_feature[text] for text in texts]
                feature_dict[category] = torch.stack(feature_list)
            else:
                feature_dict[category] = torch.empty(0, self.embed_dim)
        
        # Save to disk (only if cache_dir is configured)
        if cache_file is not None:
            try:
                # Create directory if needed
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(feature_dict, f)
                logger.info(f"Saved CLIP features to {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save cache file {cache_file}: {e}")
        
        return feature_dict
    
    def load_cached_features(
        self, 
        dataset_type: str, 
        subset_id: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Load cached features from disk if available.
        
        Args:
            dataset_type: Type of dataset
            subset_id: Identifier for subset
            
        Returns:
            Dictionary of features or None if not cached
        """
        cache_file = self._get_cache_filename(dataset_type, subset_id)

        if cache_file is not None and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        return None
    
    def get_feature_for_text(self, text: str, use_cache: bool = True) -> torch.Tensor:
        """
        Get CLIP feature for a single text.
        
        Args:
            text: Text to encode
            use_cache: Whether to use caching
            
        Returns:
            CLIP embedding tensor
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            if cache_key in self.memory_cache:
                self.memory_cache.move_to_end(cache_key)
                return self.memory_cache[cache_key]
        
        # Encode text
        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            embedding = self.clip_model.encode_text(tokens).cpu().float()[0]
        
        if use_cache:
            self._add_to_memory_cache(cache_key, embedding)
        
        return embedding
    
    def clear_memory_cache(self):
        """Clear in-memory cache."""
        self.memory_cache.clear()
        self.cache_size_bytes = 0
        logger.info("Cleared in-memory CLIP feature cache")