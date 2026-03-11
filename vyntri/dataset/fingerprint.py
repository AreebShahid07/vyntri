import os
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image
import numpy as np
from ..core.config import Config

class Fingerprinter:
    """Computes statistical and visual fingerprints of a dataset."""

    def __init__(self, config: Config):
        self.config = config

    def compute_stats(self, dataset_path: str) -> Dict[str, Any]:
        """Computes basic statistics of the dataset."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        class_counts = Counter()
        resolutions = []
        total_images = 0
        
        # Iterate through classes (subdirectories)
        # Assuming folder-per-class structure
        classes = [d for d in path.iterdir() if d.is_dir()]
        
        if not classes:
            raise ValueError(f"No class subdirectories found in {dataset_path}. Expected folder-per-class structure.")
        
        for class_dir in classes:
            class_name = class_dir.name
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
            count = len(images)
            class_counts[class_name] = count
            total_images += count
            
            # Sample resolutions (limit to config sample size for speed)
            for img_file in images[:min(len(images), 50)]: # Check first 50 per class for speed
                 try:
                    with Image.open(img_file) as img:
                        resolutions.append(img.size)
                 except Exception:
                     pass

        num_classes = len(classes)
        
        # Resolution stats
        if resolutions:
            widths = [r[0] for r in resolutions]
            heights = [r[1] for r in resolutions]
            res_mean = (np.mean(widths), np.mean(heights))
            res_std = (np.std(widths), np.std(heights))
        else:
            res_mean = (0, 0)
            res_std = (0, 0)

        # Class balance
        if total_images > 0 and num_classes > 0:
            expected_count = total_images / num_classes
            # Simple imbalance metric (0 = perfectly balanced, higher = imbalanced)
            imbalance_score = np.std(list(class_counts.values())) / expected_count if expected_count > 0 else 0
        else:
            imbalance_score = 0

        return {
            "num_classes": num_classes,
            "total_samples": total_images,
            "class_counts": dict(class_counts),
            "resolution_mean": res_mean,
            "resolution_std": res_std,
            "imbalance_score": imbalance_score
        }

    def compute_visual_fingerprint(self, dataset_path: str) -> Dict[str, Any]:
        """
        Computes visual fingerprint (texture, geometrical properties).
        For Phase 1, this is a placeholder returning dummy complexity scores.
        """
        # TODO: Implement actual visual feature extraction (Spectral, Texture, etc.)
        return {
            "spectral_signature": "pending",
            "texture_complexity": 0.5, # Placeholder
            "geometric_consistency": 0.8 # Placeholder
        }
