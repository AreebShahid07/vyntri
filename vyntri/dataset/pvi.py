import numpy as np
import torch
from typing import Dict, Any, List
from sklearn.linear_model import RidgeClassifierCV
from tqdm import tqdm
from ..core.config import Config
from ..backbones.loader import load_backbone, get_transform
from PIL import Image
from pathlib import Path

class PVIEstimator:
    """
    Estimates Pointwise V-Information (PVI) using a real pre-trained backbone.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu_if_available else "cpu")
        self.backbone = None
        self.transform = get_transform()

    def _load_backbone(self):
        if self.backbone is None:
            print(f"Loading PVI backbone: {self.config.pvi_backbone} on {self.device}...")
            self.backbone = load_backbone(self.config.pvi_backbone).to(self.device)

    def compute_pvi(self, dataset_path: str) -> Dict[str, Any]:
        """
        Computes PVI stats for the dataset.
        """
        self._load_backbone()
        
        path = Path(dataset_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # 1. Load Data and Extract Features
        features = []
        labels = []
        class_names = sorted([d.name for d in path.iterdir() if d.is_dir()])
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        print("Extracting features for PVI estimation...")
        for class_name in class_names:
            class_dir = path / class_name
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
            
            # Limit samples if configured
            if self.config.fingerprint_sample_size:
                 # Simple sampling: take first N
                 images = images[:self.config.fingerprint_sample_size // len(class_names)]
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = self.transform(img).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        feat = self.backbone(img_t).cpu().numpy().flatten()
                        
                    features.append(feat)
                    labels.append(class_to_idx[class_name])
                except Exception as e:
                    # Skip corrupted images
                    pass
                    
        features = np.array(features)
        labels = np.array(labels)
        
        if len(labels) < 2:
            return {"error": "Not enough data for PVI"}

        if len(np.unique(labels)) < 2:
            return {"error": "Need at least 2 classes for PVI estimation"}

        # 2. Train Probe (Ridge Classifier)
        print("Training linear probe for PVI...")
        clf = RidgeClassifierCV(alphas=[0.1, 1.0, 10.0])
        clf.fit(features, labels)
        
        # 3. Compute Probabilities
        # Decision function gives distance to hyperplane. 
        # For multi-class, we use softmax on these scores as a proxy for probability
        scores = clf.decision_function(features)
        if len(scores.shape) == 1: # Binary case
            scores = np.vstack([-scores, scores]).T
            
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs_model = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        # Log prob of correct class (Natural log)
        log_p_model = np.log(probs_model[np.arange(len(labels)), labels] + 1e-10)
        
        # 4. Compute Null Probabilities (Priors)
        class_counts = np.bincount(labels)
        priors = class_counts / class_counts.sum()
        log_p_null = np.log(priors[labels] + 1e-10)
        
        # 5. PVI = log_p_model - log_p_null
        pvi_scores = log_p_model - log_p_null
        
        # 6. Aggregate Stats
        return {
            "mean_pvi": float(np.mean(pvi_scores)),
            "std_pvi": float(np.std(pvi_scores)),
            "min_pvi": float(np.min(pvi_scores)),
            "max_pvi": float(np.max(pvi_scores)),
            "p5": float(np.percentile(pvi_scores, 5)),
            "p95": float(np.percentile(pvi_scores, 95))
        }
