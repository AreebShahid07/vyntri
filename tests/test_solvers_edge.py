import unittest
import numpy as np
from vyntri.solvers.fly import FlyCL
from vyntri.solvers.wisard import WiSARD

class TestSolverEdgeCases(unittest.TestCase):

    def setUp(self):
        # Random data: 10 samples, 20 features
        self.X = np.random.randn(10, 20)
        # Binary labels
        self.y = np.array([0]*5 + [1]*5)

    def test_fly_high_sparsity(self):
        # Case: Sparsity * n_hidden < 1
        # n_features=20, expansion=10 => n_hidden=200
        # sparsity=0.001 => k=0.2 < 1. Should clamp to 1.
        clf = FlyCL(expansion_ratio=10, sparsity=0.001)
        clf.fit(self.X, self.y)
        pred = clf.predict(self.X)
        self.assertEqual(len(pred), 10)

    def test_wisard_adaptive_binarization(self):
        # Case: Features are not in 0..1 range (e.g. ReLU output 0..10+)
        # Old code used >0.5 threshold, which might be all True or all False.
        # New code uses mean.
        X_large = np.random.rand(10, 20) * 100 # Values 0..100
        
        clf = WiSARD(num_bits=4)
        clf.fit(X_large, self.y)
        pred = clf.predict(X_large)
        self.assertEqual(len(pred), 10)
        # It should at least run without error and not just crash

if __name__ == '__main__':
    unittest.main()
