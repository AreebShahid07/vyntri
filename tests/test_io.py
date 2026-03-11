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
from vyntri.core.io import save_model, load_model

class TestPersistence(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = Path(self.test_dir) / "dummy_dataset_io"
        self.dataset_path.mkdir()
        self.model_path = str(Path(self.test_dir) / "model.pkl")

        (self.dataset_path / "class_a").mkdir()
        (self.dataset_path / "class_b").mkdir()

        # Create dummy RGB images
        for i in range(5):
             self._create_dummy_image(self.dataset_path / "class_a" / f"img{i}.jpg")
             self._create_dummy_image(self.dataset_path / "class_b" / f"img{i}.jpg")
        
        self.config = Config()
        self.config.use_gpu_if_available = False
        self.pipeline = VyntriPipeline(self.config)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dummy_image(self, path):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(path)

    def test_save_and_load(self):
        # 1. Train
        result = self.pipeline.run(str(self.dataset_path))
        
        # 2. Save
        save_model(result, self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
        
        # 3. Load
        model_ctx = load_model(self.model_path)
        self.assertIn('backbone', model_ctx)
        self.assertIn('solver', model_ctx)
        self.assertIn('class_names', model_ctx)
        self.assertEqual(model_ctx['class_names'], ['class_a', 'class_b'])
        
        # 4. Predict using loaded model
        # Just check if we can run inference logic
        backbone = model_ctx['backbone']
        img_t = torch.randn(1, 3, 224, 224) # dummy input
        with torch.no_grad():
             feat = backbone(img_t).cpu().numpy().flatten()
        solver = model_ctx['solver']
        pred = solver.predict(feat.reshape(1, -1))
        self.assertIsNotNone(pred)

if __name__ == '__main__':
    unittest.main()
