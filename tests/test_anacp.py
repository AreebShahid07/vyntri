import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from vyntri.solvers.anacp import AnalyticContrastiveProjection

class TestAnaCP(unittest.TestCase):

    def test_synthetic_classification(self):
        # Generate synthetic data
        X, y = make_classification(
            n_samples=500, 
            n_features=20, 
            n_informative=10, 
            n_classes=3, 
            n_clusters_per_class=1,
            random_state=42
        )
        
        # Split train/test
        train_idx = int(0.8 * len(X))
        X_train, X_test = X[:train_idx], X[train_idx:]
        y_train, y_test = y[:train_idx], y[train_idx:]
        
        # Initialize and fit AnaCP
        model = AnalyticContrastiveProjection(lambda_reg=1e-3, projection_dim=64)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Check accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"AnaCP Accuracy on Synthetic Data: {acc:.4f}")
        
        # Expect reasonably high accuracy (> 80%)
        self.assertGreater(acc, 0.80)
        
    def test_predict_proba(self):
        # n_informative=3 is enough for 3 classes * 1 cluster per class
        X, y = make_classification(n_samples=100, n_features=20, n_informative=5, n_classes=3, random_state=42)
        model = AnalyticContrastiveProjection()
        model.fit(X, y)
        probs = model.predict_proba(X)
        
        # Check constraints
        self.assertEqual(probs.shape, (100, 3))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0))
        self.assertTrue(np.all(probs >= 0) and np.all(probs <= 1))

if __name__ == '__main__':
    unittest.main()
