import unittest
import shutil
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

from vyntri.dataset.pvi import PVIEstimator
from vyntri.core.config import Config

class TestRealPVI(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.test_dir) / "dummy_dataset_pvi"
        self.dataset_path.mkdir()

        (self.dataset_path / "class_a").mkdir()
        (self.dataset_path / "class_b").mkdir()

        # Create dummy images (must be RGB for MobileNet)
        for i in range(5):
             self._create_dummy_image(self.dataset_path / "class_a" / f"img{i}.jpg")
             self._create_dummy_image(self.dataset_path / "class_b" / f"img{i}.jpg")
        
        self.config = Config()
        # Use CPU for tests
        self.config.use_gpu_if_available = False
        self.estimator = PVIEstimator(self.config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_image(self, path):
        # Create random colorful images
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(path)

    def test_compute_pvi(self):
        # This will download the model on first run
        stats = self.estimator.compute_pvi(str(self.dataset_path))
        
        print(f"PVI Stats: {stats}")
        
        self.assertIn("mean_pvi", stats)
        self.assertIn("min_pvi", stats)
        # Check that it returns floats, not tensors or numpy arrays
        self.assertIsInstance(stats['mean_pvi'], float)

if __name__ == '__main__':
    unittest.main()
