import unittest
import shutil
import tempfile
import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from vyntri.core.pipeline import VyntriPipeline
from vyntri.core.config import Config

class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.test_dir) / "dummy_dataset_pipeline"
        self.dataset_path.mkdir()

        (self.dataset_path / "class_a").mkdir()
        (self.dataset_path / "class_b").mkdir()

        # Create dummy RGB images
        for i in range(5):
             self._create_dummy_image(self.dataset_path / "class_a" / f"img{i}.jpg")
             self._create_dummy_image(self.dataset_path / "class_b" / f"img{i}.jpg")
        
        self.config = Config()
        self.config.use_gpu_if_available = False # Force CPU for tests
        self.pipeline = VyntriPipeline(self.config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_image(self, path):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(path)

    def test_run_pipeline(self):
        result = self.pipeline.run(str(self.dataset_path))
        
        self.assertIn("selected_backbone", result)
        self.assertIn("solver", result)
        
        # Test Prediction
        # Predict on one of the proper images
        test_img_path = str(self.dataset_path / "class_a" / "img0.jpg")
        prediction = self.pipeline.predict_image(test_img_path)
        print(f"Predicted: {prediction}")
        self.assertIn(prediction, ["class_a", "class_b"])

if __name__ == '__main__':
    unittest.main()
