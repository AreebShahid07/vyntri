from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    """Global configuration for Vyntri."""
    
    # Dataset Analysis Settings
    max_image_resolution: int = 1024
    min_image_resolution: int = 32
    
    # Fingerprinting
    fingerprint_sample_size: int = 1000  # Number of images to sample for fingerprinting
    
    # Execution
    use_gpu_if_available: bool = False  # CPU-first philosophy
    num_workers: int = 4
    
    # PVI Estimation
    pvi_backbone: str = "mobilenet_v3_small"
    pvi_batch_size: int = 32

    # Paths
    cache_dir: str = ".vyntri_cache"
