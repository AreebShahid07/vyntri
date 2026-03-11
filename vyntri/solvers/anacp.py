import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional

class AnalyticContrastiveProjection(BaseEstimator, ClassifierMixin):
    """
    AnaCP: Analytic adaptation via contrastive projection.
    Replaces gradient descent with closed-form linear algebra.
    """

    def __init__(self, lambda_reg: float = 1e-4, projection_dim: int = 128):
        self.lambda_reg = lambda_reg
        self.projection_dim = projection_dim
        self.W_cp: Optional[np.ndarray] = None  # Contrastive projection matrix
        self.W_clf: Optional[np.ndarray] = None  # Classifier weights
        self.classes_: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray, labels: np.ndarray, sample_weights: Optional[np.ndarray] = None):
        """Fit model analytically (no gradient descent). Supports L3A via sample_weights."""
        unique = np.unique(labels)
        if len(unique) < 2:
            raise ValueError(f"Need at least 2 classes to fit, got {len(unique)}.")
        
        # Step 1: Compute contrastive projection (Standard LDA for now, or Weighted LDA?)
        # For simplicity in Phase 7, we keep LDA standard, but apply weights to Ridge Classifier.
        self.W_cp = self._compute_contrastive_projection(features, labels)
        
        # Step 2: Project features
        features_projected = features @ self.W_cp
        
        # Step 3: Ridge regression classifier (Weighted if weights provided)
        self.W_clf = self._ridge_regression(features_projected, labels, sample_weights)
        
        # Store classes for later lookup
        self.classes_ = np.unique(labels)
        
        return self

    def _compute_contrastive_projection(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute projection matrix via Fisher LDA."""
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        n_features = features.shape[1]
        
        # Global mean
        mean_global = features.mean(axis=0)
        
        # Optimization: Use decomposition S_W = (X^T X) - sum(n_c * mu_c * mu_c^T)
        # This avoids creating centered copies of X_c which doubles memory.
        
        # 1. Total Scatter (Raw/Uncentered)
        # Note: If features is huge, this is the heaviest step, but standard BLAS handles distinct matrices.
        S_T_raw = features.T @ features
        
        # 2. Accumulate corrections
        S_W_correction = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for c in unique_classes:
            # We still need mean of class c.
            # features[labels == c] creates a temporary copy. 
            # Ideally we'd avoid this too, but it's smaller than (X_c - mean).
            mask = (labels == c)
            X_c_subset = features[mask]
            n_c = X_c_subset.shape[0]
            
            if n_c == 0: continue
            
            mean_c = X_c_subset.mean(axis=0)
            
            # S_W correction: sum(n_c * mu_c * mu_c^T)
            mean_c_col = mean_c.reshape(-1, 1)
            term = n_c * (mean_c_col @ mean_c_col.T)
            S_W_correction += term
            
            # S_B: sum(n_c * (mu_c - mu_g)(mu_c - mu_g)^T)
            diff = (mean_c - mean_global).reshape(-1, 1)
            S_B += n_c * (diff @ diff.T)
            
        # 3. Final S_W
        S_W = S_T_raw - S_W_correction
            
        # Regularize S_W to ensure distinct eigenvalues and invertibility
        S_W += self.lambda_reg * np.eye(n_features)
        
        # S_B and S_W are symmetric. detailed derivation shows we can use eigh
        # for generalized eigenvalue problem S_B v = lambda S_W v
        import scipy.linalg
        try:
            # eigvals are returned in ascending order by eigh
            eigvals, eigvecs = scipy.linalg.eigh(S_B, S_W)
            
            # Select top-k eigenvectors (reverse to get largest)
            k = min(self.projection_dim, n_classes - 1, n_features)
            idx = np.arange(len(eigvals) - 1, len(eigvals) - 1 - k, -1)
            W_cp = eigvecs[:, idx]
            
            # Normalize
            W_cp /= np.linalg.norm(W_cp, axis=0, keepdims=True) + 1e-10
            
            return W_cp
            
        except np.linalg.LinAlgError:
            # Fallback
            return np.eye(n_features, min(self.projection_dim, n_features))

    def _ridge_regression(
        self, 
        features: np.ndarray, 
        labels: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Solve ridge regression analytically. weighted if sample_weights provided (L3A)."""
        
        # One-hot encode labels
        unique_classes = np.unique(labels)
        n_classes = len(unique_classes)
        label_map = {l: i for i, l in enumerate(unique_classes)}
        y_indices = np.array([label_map[l] for l in labels])
        Y = np.eye(n_classes)[y_indices]
        
        n_samples = features.shape[0]
        
        if sample_weights is None:
            # Standard: W* = (X^T X + lambda I)^-1 X^T Y
            G = features.T @ features + self.lambda_reg * np.eye(features.shape[1])
            C = features.T @ Y
        else:
            # Weighted: W* = (X^T W X + lambda I)^-1 X^T W Y
            # W is diagonal matrix of weights.
            # Efficiently: X^T W X = (X * sqrt(w))^T (X * sqrt(w))
            # Or just numpy broadcasting: (features.T * weights) @ features
            
            # W_diag = np.diag(sample_weights) # Too big for memory!
            # Use broadcasting:
            
            # X_weighted = W @ X -> conceptually.
            # G = X^T @ W @ X
            # C = X^T @ W @ Y
            
            # X.T @ (weights[:, None] * X)
            G = features.T @ (sample_weights[:, None] * features) 
            G += self.lambda_reg * np.eye(features.shape[1])
            
            C = features.T @ (sample_weights[:, None] * Y)
            
        # Solve: W* = G^-1 C
        W = np.linalg.solve(G, C)
        
        return W

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.classes_ is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        probs = self.predict_proba(features)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.W_cp is None or self.W_clf is None:
            raise RuntimeError("Model not fitted yet.")
            
        features_projected = features @ self.W_cp
        logits = features_projected @ self.W_clf
        
        # Softmax
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs
