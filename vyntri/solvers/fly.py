import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifier

class FlyCL(BaseEstimator, ClassifierMixin):
    """
    Fly-CL: Bio-Inspired Sparse Coding.
     mimics the fruit fly olfactory circuit.
     Projects features into high-dim sparse space using random binary weights + WTA.
    """
    
    def __init__(self, expansion_ratio: int = 10, sparsity: float = 0.1, ridge_alpha: float = 1.0, random_state: int = None):
        self.expansion_ratio = expansion_ratio
        self.sparsity = sparsity
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.W_rand = None
        self.clf = RidgeClassifier(alpha=ridge_alpha)
        
    def fit(self, X, y):
        n_features = X.shape[1]
        n_hidden = int(n_features * self.expansion_ratio)
        
        # 1. Random Projection Layer (Input -> Kenyon Cells)
        rng = np.random.RandomState(self.random_state)
        self.W_rand = rng.randn(n_features, n_hidden)
        
        # 2. Project and Apply WTA (Winner-Take-All)
        X_sparse = self._transform_hidden(X)
        
        # 3. Train Readout (Kenyon Cells -> Output)
        # Ridge Regression is biologically plausible (Hebbian-like).
        self.clf.fit(X_sparse, y)
        self.classes_ = self.clf.classes_
        return self
        
    def predict(self, X):
        if self.W_rand is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        X_sparse = self._transform_hidden(X)
        return self.clf.predict(X_sparse)
        
    def _transform_hidden(self, X):
        # Linear projection
        activations = X @ self.W_rand
        
        # Winner-Take-All (Sparsity)
        # Keep top k activations, set others to 0.
        # Return binary 1/0? Or keep values?
        # Bio-FO often uses binary. Let's try binary.
        
        k = max(1, int(activations.shape[1] * self.sparsity))
        # Find threshold per sample
        # np.partition is fast
        thresholds = np.partition(activations, -k, axis=1)[:, -k]
        
        # Binary sparse representation
        mask = activations >= thresholds[:, None]
        return mask.astype(float)
