from typing import Dict, Any, Tuple
from ..core.config import Config
from ..dataset.engine import DatasetFingerprint

class ModelSelector:
    """
    Selects the optimal backbone based on dataset fingerprint.
    """

    def __init__(self, config: Config):
        self.config = config

    def select(self, fingerprint: Dict[str, Any]) -> str:
        """
        Selects backbone.
        Strategy for Phase 3:
        - If 'mean_pvi' implies 'hard' dataset (low PVI, e.g. < 0.5) OR high resolution -> use ResNet18
        - Else -> use MobileNetV3 (faster)
        """
        stats = fingerprint['stats']
        complexity = fingerprint.get('complexity', {})
        pvi_stats = complexity.get('pvi', {})
        
        # Check PVI (if available and not error)
        mean_pvi = pvi_stats.get('mean_pvi')
        
        if isinstance(mean_pvi, float):
             # Lower PVI means model is less confident relative to prior -> Harder task
             if mean_pvi < 0.5:
                 return "resnet18"
                 
        # Check Resolution
        res_mean = stats.get('resolution_mean', (0,0))
        # If avg resolution > 224x224 significantly, maybe bigger model helps? 
        # Actually MobileNet is fine for 224. ResNet is also 224 by default transform.
        # Let's say if we have MANY classes, use ResNet.
        
        num_classes = stats.get('num_classes', 0)
        if num_classes > 50:
            return "resnet18"

        # Default to lightweight
        return "mobilenet_v3_small"
