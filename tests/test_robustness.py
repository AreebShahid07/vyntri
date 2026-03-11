"""
Robustness and edge-case tests for vyntri.
Covers: predict-before-fit, single class, empty data, mismatched features,
very small datasets, many-class scenarios, string labels, and solver stability.
"""
import unittest
import shutil
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.datasets import make_classification

from vyntri.solvers.anacp import AnalyticContrastiveProjection
from vyntri.solvers.continual import ContinualAnaCP
from vyntri.solvers.fly import FlyCL
from vyntri.solvers.wisard import WiSARD
from vyntri.dataset.fingerprint import Fingerprinter
from vyntri.core.config import Config


class TestPredictBeforeFit(unittest.TestCase):
    """All solvers must give a clear error when predict is called before fit."""

    def test_anacp_predict_before_fit(self):
        model = AnalyticContrastiveProjection()
        with self.assertRaises(RuntimeError):
            model.predict(np.random.randn(5, 10))

    def test_anacp_predict_proba_before_fit(self):
        model = AnalyticContrastiveProjection()
        with self.assertRaises(RuntimeError):
            model.predict_proba(np.random.randn(5, 10))

    def test_fly_predict_before_fit(self):
        model = FlyCL()
        with self.assertRaises(RuntimeError):
            model.predict(np.random.randn(5, 10))

    def test_wisard_predict_before_fit(self):
        model = WiSARD()
        with self.assertRaises(RuntimeError):
            model.predict(np.random.randn(5, 10))


class TestSingleClassGuard(unittest.TestCase):
    """AnaCP must reject single-class data instead of crashing with LinAlgError."""

    def test_anacp_single_class(self):
        X = np.random.randn(20, 10)
        y = np.zeros(20)
        model = AnalyticContrastiveProjection()
        with self.assertRaises(ValueError):
            model.fit(X, y)

    def test_continual_single_class(self):
        X = np.random.randn(20, 10)
        y = np.array(["cat"] * 20)
        model = ContinualAnaCP()
        with self.assertRaises(ValueError):
            model.fit(X, y)


class TestStringLabels(unittest.TestCase):
    """Solvers must work with string labels, not just integer labels."""

    def test_anacp_string_labels(self):
        X, y_int = make_classification(n_samples=100, n_features=20, n_informative=5,
                                       n_classes=3, random_state=42)
        y_str = np.array([f"class_{i}" for i in y_int])
        model = AnalyticContrastiveProjection()
        model.fit(X, y_str)
        preds = model.predict(X)
        self.assertTrue(all(p.startswith("class_") for p in preds))

    def test_fly_string_labels(self):
        X, y_int = make_classification(n_samples=80, n_features=20, n_informative=5,
                                       n_classes=3, random_state=42)
        y_str = np.array([f"label_{i}" for i in y_int])
        model = FlyCL(random_state=42)
        model.fit(X, y_str)
        preds = model.predict(X)
        self.assertTrue(all(p.startswith("label_") for p in preds))

    def test_wisard_string_labels(self):
        X = np.random.randint(0, 2, (100, 20))
        y = np.array(["cat"] * 50 + ["dog"] * 50)
        model = WiSARD(num_bits=4)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertTrue(all(p in ("cat", "dog") for p in preds))


class TestSmallDatasets(unittest.TestCase):
    """Solvers must not crash on minimal viable datasets (2 samples, 2 classes)."""

    def test_anacp_two_samples(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        model = AnalyticContrastiveProjection(projection_dim=1)
        model.fit(X, y)
        pred = model.predict(X)
        self.assertEqual(len(pred), 2)

    def test_fly_two_samples(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        model = FlyCL(expansion_ratio=5, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        self.assertEqual(len(pred), 2)

    def test_wisard_two_samples(self):
        X = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        y = np.array([0, 1])
        model = WiSARD(num_bits=2)
        model.fit(X, y)
        pred = model.predict(X)
        self.assertEqual(len(pred), 2)


class TestManyClasses(unittest.TestCase):
    """Solvers should work with many classes (10+)."""

    def test_anacp_10_classes(self):
        X, y = make_classification(n_samples=300, n_features=50, n_informative=30,
                                   n_classes=10, n_clusters_per_class=1, random_state=42)
        model = AnalyticContrastiveProjection(projection_dim=64)
        model.fit(X, y)
        preds = model.predict(X)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.5)

    def test_fly_10_classes(self):
        X, y = make_classification(n_samples=300, n_features=50, n_informative=30,
                                   n_classes=10, n_clusters_per_class=1, random_state=42)
        model = FlyCL(expansion_ratio=10, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        # At least better than random (10%)
        acc = np.mean(preds == y)
        self.assertGreater(acc, 0.15)


class TestContinualUpdate(unittest.TestCase):
    """ContinualAnaCP must handle sequential updates without degradation."""

    def test_multiple_updates(self):
        """Train once, then update 3 times — model must not crash or NaN."""
        X, y = make_classification(n_samples=100, n_features=20, n_informative=5,
                                   n_classes=3, random_state=42)
        y_str = np.array([f"c{i}" for i in y])
        model = ContinualAnaCP(projection_dim=10)
        model.fit(X, y_str)

        for batch_idx in range(3):
            X_new = np.random.randn(20, 20)
            y_new = np.array([f"c{i % 3}" for i in range(20)])
            model.update(X_new, y_new)

        preds = model.predict(X)
        self.assertEqual(len(preds), 100)
        self.assertFalse(any(p is None for p in preds))

    def test_update_adds_new_classes(self):
        """Update must gracefully expand to new classes."""
        X = np.random.randn(40, 10)
        y = np.array(["a"] * 20 + ["b"] * 20)
        model = ContinualAnaCP(projection_dim=5)
        model.fit(X, y)
        self.assertEqual(len(model.classes_), 2)

        X_new = np.random.randn(20, 10)
        y_new = np.array(["c"] * 10 + ["d"] * 10)
        model.update(X_new, y_new)
        self.assertEqual(len(model.classes_), 4)
        self.assertIn("c", model.classes_)
        self.assertIn("d", model.classes_)


class TestFlyCLReproducibility(unittest.TestCase):
    """FlyCL with random_state must produce identical results."""

    def test_reproducible(self):
        X, y = make_classification(n_samples=50, n_features=20, n_informative=5,
                                   n_classes=2, random_state=42)
        m1 = FlyCL(random_state=123)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = FlyCL(random_state=123)
        m2.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_differ(self):
        X, y = make_classification(n_samples=50, n_features=20, n_informative=5,
                                   n_classes=2, random_state=42)
        m1 = FlyCL(random_state=1)
        m1.fit(X, y)

        m2 = FlyCL(random_state=999)
        m2.fit(X, y)

        # Random projections should differ
        self.assertFalse(np.array_equal(m1.W_rand, m2.W_rand))


class TestEmptyDataset(unittest.TestCase):
    """Fingerprinter and pipeline must fail clearly on empty/malformed datasets."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fingerprint_no_subdirs(self):
        # Dataset dir exists but has no class subdirectories
        dataset = Path(self.test_dir) / "empty_dataset"
        dataset.mkdir()
        fp = Fingerprinter(Config())
        with self.assertRaises(ValueError):
            fp.compute_stats(str(dataset))

    def test_fingerprint_nonexistent_path(self):
        fp = Fingerprinter(Config())
        with self.assertRaises(FileNotFoundError):
            fp.compute_stats("/nonexistent/path/xyz")

    def test_fingerprint_empty_class_dirs(self):
        # Class dirs exist but contain no images
        dataset = Path(self.test_dir) / "no_images"
        dataset.mkdir()
        (dataset / "class_a").mkdir()
        (dataset / "class_b").mkdir()
        fp = Fingerprinter(Config())
        stats = fp.compute_stats(str(dataset))
        self.assertEqual(stats["total_samples"], 0)


class TestSolverNumericalStability(unittest.TestCase):
    """Test solvers with adversarial numerical conditions."""

    def test_anacp_identical_features(self):
        """All features identical except labels — solver should not NaN."""
        X = np.ones((20, 10))
        # Add tiny noise to prevent perfect degeneracy
        X += np.random.randn(20, 10) * 1e-8
        y = np.array([0] * 10 + [1] * 10)
        model = AnalyticContrastiveProjection(lambda_reg=1e-2)
        model.fit(X, y)
        probs = model.predict_proba(X)
        self.assertFalse(np.any(np.isnan(probs)))

    def test_anacp_high_dimensional(self):
        """More features than samples (p >> n) — ridge regularization should handle this."""
        X = np.random.randn(10, 200)
        y = np.array([0] * 5 + [1] * 5)
        model = AnalyticContrastiveProjection(lambda_reg=1.0, projection_dim=1)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), 10)
        self.assertFalse(np.any(np.isnan(model.predict_proba(X))))

    def test_wisard_single_feature(self):
        """Extreme case: only 1 feature per sample."""
        X = np.array([[0], [1], [0], [1], [0], [1]])
        y = np.array([0, 1, 0, 1, 0, 1])
        model = WiSARD(num_bits=1)
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), 6)


class TestConfigDefaults(unittest.TestCase):
    """Config default values must be sane."""

    def test_defaults(self):
        c = Config()
        self.assertFalse(c.use_gpu_if_available)
        self.assertGreater(c.fingerprint_sample_size, 0)
        self.assertEqual(c.pvi_backbone, "mobilenet_v3_small")
        self.assertGreater(c.num_workers, 0)

    def test_custom_config(self):
        c = Config(use_gpu_if_available=True, fingerprint_sample_size=500)
        self.assertTrue(c.use_gpu_if_available)
        self.assertEqual(c.fingerprint_sample_size, 500)


if __name__ == '__main__':
    unittest.main()
