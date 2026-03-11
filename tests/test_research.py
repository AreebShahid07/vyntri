import unittest
import numpy as np
from sklearn.datasets import make_classification

from vyntri.solvers.anacp import AnalyticContrastiveProjection
from vyntri.solvers.fly import FlyCL
from vyntri.solvers.wisard import WiSARD

class TestResearchSolvers(unittest.TestCase):

    def test_l3a_weighted_anacp(self):
        print("\nTesting L3A (Weighted AnaCP)...")
        # Create imbalanced dataset
        X, y = make_classification(n_samples=200, n_features=20, n_classes=2, weights=[0.9, 0.1], random_state=42)
        
        # Calculate weights (Inverse Frequency)
        classes, counts = np.unique(y, return_counts=True)
        weight_map = {c: 1.0/cnt for c, cnt in zip(classes, counts)}
        sample_weights = np.array([weight_map[label] for label in y])
        
        # Normalize weights
        sample_weights = sample_weights / sample_weights.mean()
        
        model = AnalyticContrastiveProjection()
        model.fit(X, y, sample_weights=sample_weights)
        acc = model.score(X, y)
        print(f"L3A Accuracy: {acc:.4f}")
        self.assertGreater(acc, 0.70)

    def test_fly_cl(self):
        print("\nTesting Fly-CL...")
        # Fix: n_informative=5 to allow 3 classes * 2 clusters
        X, y = make_classification(n_samples=100, n_features=20, n_informative=5, n_classes=3, random_state=42)
        model = FlyCL(expansion_ratio=10, sparsity=0.1) # Increased expansion
        model.fit(X, y)
        acc = model.score(X, y)
        print(f"Fly-CL Accuracy: {acc:.4f}")
        self.assertGreater(acc, 0.60)

    def test_wisard(self):
        print("\nTesting WiSARD (Sanity Check)...")
        # Trivial task: Class is determined essentially by the first bit/feature
        # If WiSARD can't learn this, the mapping logic is broken.
        X = np.random.randint(0, 2, (1000, 20))
        y = X[:, 0] # Class = first feature
        
        # Split train/test
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        # num_bits=1 or more should solve this if mapped correctly
        # random mapping MIGHT miss feature 0 in the first RAM if strictly partitioned?
        # But we partition by input index. Feature 0 will go to SOME RAM.
        # RAMs outputs are summed.
        # If feature 0 is 1, the RAM containing feature 0 will output a specific address.
        # The discriminator for Class 1 will have that address set.
        # The discriminator for Class 0 will NOT (or will have address for feature 0=0).
        # So it should be 100% separable.
        
        model = WiSARD(num_bits=4) 
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = np.mean(preds == y_test)
        print(f"WiSARD Accuracy: {acc:.4f}")
        self.assertGreater(acc, 0.90)

if __name__ == '__main__':
    unittest.main()
