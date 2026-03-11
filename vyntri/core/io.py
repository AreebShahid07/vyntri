import torch
from pathlib import Path
from typing import Any, Dict
from ..backbones.loader import load_backbone

def save_model(pipeline_result: Dict[str, Any], path: str):
    """
    Saves the trained model components to a single file using torch.save.
    """
    data = {
        "selected_backbone": pipeline_result["selected_backbone"],
        # We pickle the solver (which contains the numpy matrices)
        "solver": pipeline_result["solver"],
        "fingerprint": pipeline_result.get("fingerprint"),
        "classes": pipeline_result["solver"].classes_,
        "class_names": pipeline_result.get("class_names")
    }
    
    torch.save(data, path)
    print(f"Model saved to {path} (using torch.save)")

def load_model(path: str, config=None) -> Dict[str, Any]:
    """
    Loads a model and reconstructs the necessary components for inference.
    Returns a dictionary acting as a lightweight context.
    WARNING: uses torch.load which relies on pickle. Only load models from trusted sources.
    """
    # We must set weights_only=False to load custom solver objects/classes.
    data = torch.load(path, weights_only=False)
        
    backbone_name = data["selected_backbone"]
    
    # Load backbone (fresh copy)
    backbone = load_backbone(backbone_name)
    
    return {
        "backbone": backbone,
        "backbone_name": backbone_name, # Storing this so we know what to save back
        "solver": data["solver"],
        "classes": data["classes"],
        "fingerprint": data.get("fingerprint"),
        "class_names": data.get("class_names")
    }
