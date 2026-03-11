import unittest
import shutil
import tempfile
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

from vyntri.core.pipeline import VyntriPipeline
from vyntri.core.config import Config
from vyntri.core.io import save_model, load_model
from vyntri.solvers.continual import ContinualAnaCP

class TestContinualLearning(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.dataset_a = Path(self.test_dir) / "dataset_a"
        self.dataset_b = Path(self.test_dir) / "dataset_b"
        self.model_path = str(Path(self.test_dir) / "continual_model.pkl")

        # Create two datasets with DIFFERENT classes
        self._create_dataset(self.dataset_a, ["cat", "dog"])
        self._create_dataset(self.dataset_b, ["bird", "fish"])
        
        self.config = Config()
        self.config.use_gpu_if_available = False # CPU for test

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _create_dataset(self, path, classes):
        path.mkdir()
        for cls in classes:
            (path / cls).mkdir()
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(path / cls / f"img{i}.jpg")

    def test_incremental_learning(self):
        # 1. Train on Dataset A (Cat vs Dog)
        print("Training on Task A...")
        pipeline = VyntriPipeline(self.config)
        result_a = pipeline.run(str(self.dataset_a))
        
        # Verify it used ContinualAnaCP
        self.assertIsInstance(result_a['solver'], ContinualAnaCP)
        self.assertTrue(hasattr(result_a['solver'], 'G'))
        
        # Save model
        save_model(result_a, self.model_path)
        
        # 2. Load and Update on Dataset B (Bird vs Fish)
        print("Updating on Task B...")
        model_ctx = load_model(self.model_path)
        solver = model_ctx['solver']
        
        # Simulate extraction pipeline (simplified)
        pipeline.model_backbone = model_ctx['backbone']
        pipeline.device = pipeline.device
        features_b, labels_b, class_names_b = pipeline._extract_features(str(self.dataset_b))
        
        real_labels_b = np.array([class_names_b[i] for i in labels_b])
        
        # Verify classes before update
        self.assertEqual(len(solver.classes_), 2) # cat, dog
        
        # Update
        solver.update(features_b, real_labels_b)
        
        # Verify classes after update
        self.assertEqual(len(solver.classes_), 4) # cat, dog, bird, fish
        self.assertIn("bird", solver.classes_)
        
        # 3. Save updated model
        result_b = {
            "selected_backbone": model_ctx['backbone_name'],
            "solver": solver,
            "fingerprint": None,
            "num_samples": 0,
            "class_names": solver.classes_.tolist()
        }
        save_model(result_b, self.model_path)
        
        # 4. Final Load & Predict
        final_ctx = load_model(self.model_path)
        final_solver = final_ctx['solver']
        self.assertEqual(len(final_solver.classes_), 4)
        
        print("Continual Learning Test Passed!")

if __name__ == '__main__':
    unittest.main()
