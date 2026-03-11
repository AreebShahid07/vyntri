import unittest
import shutil
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np

from vyntri.dataset.engine import DatasetIntelligenceEngine
from vyntri.core.config import Config

class TestDatasetIntelligenceEngine(unittest.TestCase):

    def setUp(self):
        # Create a temporary dataset directory
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.test_dir) / "dummy_dataset"
        self.dataset_path.mkdir()

        # Create dummy classes
        (self.dataset_path / "class_a").mkdir()
        (self.dataset_path / "class_b").mkdir()

        # Create dummy images
        self._create_dummy_image(self.dataset_path / "class_a" / "img1.jpg")
        self._create_dummy_image(self.dataset_path / "class_a" / "img2.jpg")
        self._create_dummy_image(self.dataset_path / "class_b" / "img1.jpg")
        
        self.config = Config()
        self.engine = DatasetIntelligenceEngine(self.config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_image(self, path):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        img.save(path)

    def test_analyze(self):
        fingerprint = self.engine.analyze(str(self.dataset_path))
        
        # Check basic stats
        stats = fingerprint['stats']
        self.assertEqual(stats['num_classes'], 2)
        self.assertEqual(stats['total_samples'], 3)
        self.assertEqual(stats['class_counts']['class_a'], 2)
        self.assertEqual(stats['class_counts']['class_b'], 1)
        
        # Check resolution stats
        self.assertEqual(stats['resolution_mean'], (100.0, 100.0))
        
        # Check structure
        self.assertIn('visual', fingerprint)
        self.assertIn('complexity', fingerprint)
        self.assertIn('quality', fingerprint)

if __name__ == '__main__':
    unittest.main()
