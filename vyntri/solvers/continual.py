import numpy as np
from typing import Optional
from .anacp import AnalyticContrastiveProjection

class ContinualAnaCP(AnalyticContrastiveProjection):
    """
    AnaCP with recursive updates for continual learning.
    Allows adding new data/classes without retraining from scratch.
    """

    def __init__(self, lambda_reg: float = 1e-4, projection_dim: int = 128):
        super().__init__(lambda_reg, projection_dim)
        self.G: Optional[np.ndarray] = None  # Accumulated Gram matrix
        self.C: Optional[np.ndarray] = None  # Accumulated Cross-correlation matrix
        self.n_seen: int = 0  # Total samples seen

    def fit(self, features: np.ndarray, labels: np.ndarray):
        """
        Initial fit. Same as base class, but initializes recursive matrices.
        """
        super().fit(features, labels)
        
        # Initialize G and C for future updates
        # We need to re-compute G and C based on projected features
        if self.W_cp is None:
             raise RuntimeError("W_cp not initialized")
             
        features_projected = features @ self.W_cp
        
        # One-hot encode
        n_classes = len(self.classes_)
        label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        y_indices = np.array([label_to_idx[y] for y in labels])
        Y = np.eye(n_classes)[y_indices]
        
        self.G = features_projected.T @ features_projected
        self.C = features_projected.T @ Y
        self.n_seen = len(labels)
        
        return self

    def update(self, features: np.ndarray, labels: np.ndarray):
        """
        Incrementally update model with new data.
        """
        if self.W_cp is None:
            # First time seeing data, just fit
            self.fit(features, labels)
            return

        # Project features using EXISTING projection matrix
        # Note: In a true continual setting, W_cp might also need updating, 
        # but for this version we assume the initial projection is robust enough 
        # or we freeze the feature space.
        features_projected = features @ self.W_cp
        
        # Handle New Classes
        unique_new_labels = np.unique(labels)
        new_classes = np.setdiff1d(unique_new_labels, self.classes_)
        
        if len(new_classes) > 0:
            # Expand classes_ array
            self.classes_ = np.concatenate([self.classes_, new_classes])
            # Expand C matrix (add columns for new classes)
            padding = len(self.classes_) - self.C.shape[1]
            self.C = np.pad(self.C, ((0, 0), (0, padding)))
            
        # One-hot encode new batch
        label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        y_indices = np.array([label_to_idx[y] for y in labels])
        n_total_classes = len(self.classes_)
        Y_new = np.eye(n_total_classes)[y_indices]
        
        # Recursive Updates: G_new = G_old + X^T X
        self.G += features_projected.T @ features_projected
        
        # C_new = C_old + X^T Y
        # Ensure Y_new matches C's current width (it should via n_total_classes)
        self.C += features_projected.T @ Y_new
        
        # Re-solve for weights: W = (G + lambda*I)^-1 C
        G_reg = self.G + self.lambda_reg * np.eye(self.G.shape[0])
        self.W_clf = np.linalg.solve(G_reg, self.C)
        
        self.n_seen += len(labels)
