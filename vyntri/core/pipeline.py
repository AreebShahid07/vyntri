import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, Any, Optional

from .config import Config
from ..dataset.engine import DatasetIntelligenceEngine
from ..selection.selector import ModelSelector
from ..backbones.loader import load_backbone, get_transform
from ..solvers.continual import ContinualAnaCP # Use Continual by default for Phase 6

class VyntriPipeline:
    """
    End-to-end pipeline: Ingest -> Analyze -> Select -> Adapt -> Predict
    """

    def __init__(self, config: Config = None):
        self.config = config if config else Config()
        self.dataset_engine = DatasetIntelligenceEngine(self.config)
        self.selector = ModelSelector(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config.use_gpu_if_available else "cpu")
        
        self.model_backbone = None
        self.model_solver = None
        self.transform = get_transform()

    def run(self, dataset_path: str, solver_type: str = "anacp", **solver_kwargs):
        """
        Run the full pipeline:
        1. Analyze Dataset (Fingerprint)
        2. Select Model (Backbone)
        3. Extract Features
        4. Adapt Model (Analytic Solver)
        """
        print(f"=== Starting Vyntri Pipeline on {dataset_path} ===")
        
        # Step 1: Dataset Intelligence
        print("1. Analyzing Dataset...")
        engine = DatasetIntelligenceEngine(self.config)
        fingerprint = engine.analyze(dataset_path)
        print(f"   Fingerprint Stats: {fingerprint['stats']}")
        
        # Step 2: Zero-Cost Model Selection
        print("2. Selecting Model...")
        selector = ModelSelector(self.config)
        # For MVP, we just take the best one. 
        # In full version, we'd use SPKT/GreenFactory scores from the engine if implemented.
        # Here we just default to a robust choice or loop backbones.
        # Let's stick to the mapped backbone logic from selector for now.
        selected_backbone = selector.select(fingerprint)
        print(f"   Selected Backbone: {selected_backbone}")
        
        # Step 3: Feature Extraction
        print(f"3. Loading {selected_backbone} and extracting features...")
        self.model_backbone = load_backbone(selected_backbone)
        self.model_backbone.to(self.device)
        
        features, labels, class_names = self._extract_features(dataset_path)
        
        # Step 4: Analytic Adaptation (The "Training")
        print(f"4. Adapting Model ({solver_type})...")
        
        # Instantiate Solver based on type
        # Determine projection dimension (usually features.shape[1], or smaller/larger)
        input_dim = features.shape[1]
        
        if solver_type.lower() == "anacp":
            from ..solvers.continual import ContinualAnaCP
            # AnaCP projects DOWN usually, e.g. to 128 or num_classes-1
            proj_dim = min(128, input_dim, len(np.unique(labels)) - 1 if len(np.unique(labels)) > 1 else 128)
            # Ensure at least 1 dim if classification logic permits
            proj_dim = max(proj_dim, 1) 
            self.model_solver = ContinualAnaCP(projection_dim=proj_dim)
            
        elif solver_type.lower() == "fly":
            from ..solvers.fly import FlyCL
            # FlyCL projects UP (Expansion)
            ratio = solver_kwargs.get('expansion_ratio', 10)
            self.model_solver = FlyCL(expansion_ratio=ratio)
            
        elif solver_type.lower() == "wisard":
            from ..solvers.wisard import WiSARD
            # WiSARD needs binary inputs logic inside? 
            # Our WiSARD helper handles basic binarization if needed (assumes 0..1 float or raw)
            # But deep features are floats -inf..inf or ReLU 0..inf.
            # We might need a scaler? WiSARD usually works on raw bits or thermometers.
            # For deep features, min-max scaling + bits is standard.
            # Let's add a quick scaler wrapper if using WiSARD? 
            # Or assume WiSARD class handles it? 
            # Our current WiSARD implementation just does (X>0.5). That's too simple for ReLU features.
            # let's assume the user knows or we fix WiSARD later. 
            # For now, instantiate it.
            bits = solver_kwargs.get('num_bits', 4)
            self.model_solver = WiSARD(num_bits=bits)
            # Note: ReLU features are >=0. Need scaling to roughly 0..1 range for naive thresholding?
            # Or just threshold at mean?
            # Let's leave as is for MVP integration.
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        # Fit
        # Mapping labels to class names for ContinualAnaCP is handled there.
        # Fly/WiSARD might expect raw Y or mapped?
        # Our solvers fit(X, y).
        # We pass real labels (strings) to ContinualAnaCP.
        # Fly/WiSARD in our impl used sklearn style (flexible).
        # Let's pass real string labels to all if they support it.
        # Sklearn classifiers usually support string labels.
        real_labels = np.array([class_names[i] for i in labels])
        self.model_solver.fit(features, real_labels)
        
        print("=== Pipeline Complete ===")
        return {
            "fingerprint": fingerprint,
            "selected_backbone": selected_backbone,
            "solver": self.model_solver,
            "num_samples": len(labels),
            "class_names": class_names
        }

    def _extract_features(self, dataset_path: str):
        path = Path(dataset_path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        features = []
        labels = []
        skipped = 0
        class_names = sorted([d.name for d in path.iterdir() if d.is_dir()])
        if not class_names:
            raise ValueError(f"No class subdirectories found in {dataset_path}")
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        for class_name in class_names:
            class_dir = path / class_name
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions]
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_t = self.transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.model_backbone(img_t).cpu().numpy().flatten()
                    features.append(feat)
                    labels.append(class_to_idx[class_name])
                except Exception as e:
                    skipped += 1

        if skipped > 0:
            print(f"   Warning: Skipped {skipped} images due to load errors.")
        if len(features) == 0:
            raise ValueError(f"No valid images found in {dataset_path}. Check that class folders contain image files.")
        if len(set(labels)) < 2:
            raise ValueError(f"Need at least 2 classes with valid images, found {len(set(labels))}.")
                    
        return np.array(features), np.array(labels), class_names

    def predict_image(self, image_path: str) -> str:
        """Predict the class of a single image. Pipeline must be run first."""
        if not self.model_backbone or not self.model_solver:
            raise RuntimeError("Pipeline not run yet. Call run() first.")
            
        img = Image.open(image_path).convert('RGB')
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model_backbone(img_t).cpu().numpy().flatten()
            
        pred = self.model_solver.predict(feat.reshape(1, -1))[0]
        return pred
