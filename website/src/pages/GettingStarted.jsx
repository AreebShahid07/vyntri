import CodeBlock from '../components/CodeBlock'

export default function GettingStarted() {
  return (
    <div className="doc-page">
      <h1>Getting Started</h1>
      <p className="lead">Get Vyntri up and running in under 5 minutes.</p>

      <section>
        <h2 id="installation">Installation</h2>
        <CodeBlock title="Terminal">pip install vyntri</CodeBlock>
        <p>This installs Vyntri and all required dependencies (PyTorch, torchvision, scikit-learn, NumPy, Pillow, tqdm).</p>

        <h3>Optional: Webcam support</h3>
        <CodeBlock title="Terminal">pip install vyntri[webcam]</CodeBlock>
        <p>Adds OpenCV for real-time webcam inference demos.</p>
      </section>

      <section>
        <h2 id="requirements">Requirements</h2>
        <ul>
          <li>Python 3.9 or higher</li>
          <li>No GPU required — works on CPU by default</li>
          <li>~500 MB disk for PyTorch + backbone weights (downloaded on first use)</li>
          <li>Tested on Windows, macOS, Linux, and Raspberry Pi OS</li>
        </ul>
      </section>

      <section>
        <h2 id="prepare-dataset">Prepare Your Dataset</h2>
        <p>Organize your images into folders where each folder name is the class label:</p>
        <CodeBlock title="Folder structure">{`my_dataset/
├── cat/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
├── dog/
│   ├── photo1.jpg
│   └── ...
└── bird/
    └── ...`}</CodeBlock>
        <p>Vyntri supports JPG, JPEG, PNG, BMP, TIFF, and WebP image formats.</p>
      </section>

      <section>
        <h2 id="first-model">Train Your First Model</h2>
        <h3>Option A: Python API</h3>
        <CodeBlock title="train.py">{`from vyntri.core.pipeline import VyntriPipeline
from vyntri.core.config import Config
from vyntri.core.io import save_model

# Initialize with default config (CPU, MobileNetV3)
pipeline = VyntriPipeline(Config())

# Run full pipeline: Analyze → Select → Extract → Adapt
result = pipeline.run("./my_dataset", solver_type="anacp")

# Save the trained model
save_model(result, "my_model.pkl")
print(f"Trained on {result['num_samples']} images")
print(f"Backbone: {result['selected_backbone']}")
print(f"Classes: {result['class_names']}")`}</CodeBlock>

        <h3>Option B: Command Line</h3>
        <CodeBlock title="Terminal">{`# Train with default solver (AnaCP)
vyntri train ./my_dataset

# Or specify a solver
vyntri train ./my_dataset --solver fly

# Save to custom path
vyntri train ./my_dataset --save_path my_model.pkl`}</CodeBlock>
      </section>

      <section>
        <h2 id="predict">Make Predictions</h2>
        <h3>Python API</h3>
        <CodeBlock title="predict.py">{`from vyntri.core.io import load_model
from vyntri.backbones.loader import get_transform
from PIL import Image
import torch
import numpy as np

# Load trained model
model = load_model("my_model.pkl")
backbone = model['backbone']
solver = model['solver']

# Process image
transform = get_transform()
img = Image.open("test_photo.jpg").convert('RGB')
img_tensor = transform(img).unsqueeze(0)

# Extract features and predict
with torch.no_grad():
    features = backbone(img_tensor).numpy().flatten()

prediction = solver.predict(features.reshape(1, -1))
print(f"Predicted class: {prediction[0]}")`}</CodeBlock>

        <h3>CLI</h3>
        <CodeBlock title="Terminal">vyntri predict test_photo.jpg my_model.pkl</CodeBlock>
      </section>

      <section>
        <h2 id="continual">Add New Data (Continual Learning)</h2>
        <p>Vyntri supports incremental updates — add new images or even new classes without retraining from scratch:</p>
        <CodeBlock title="update.py">{`from vyntri.core.io import load_model, save_model

model = load_model("my_model.pkl")
solver = model['solver']

# Prepare new features (extracted from new images)
# solver.update(new_features, new_labels)

# Save updated model
# save_model(updated_result, "my_model.pkl")`}</CodeBlock>
        <CodeBlock title="Terminal">{`# CLI: point to folder with new data
vyntri update ./new_images my_model.pkl`}</CodeBlock>
        <p>The update command uses Recursive Least Squares — no need to touch the original training data.</p>
      </section>

      <section>
        <h2 id="next-steps">Next Steps</h2>
        <ul>
          <li><strong>Explore solvers</strong> — Compare AnaCP, FlyCL, and WiSARD in the <a href="#/solvers">Solvers Guide</a></li>
          <li><strong>Full API</strong> — See all classes and methods in the <a href="#/api">API Reference</a></li>
          <li><strong>CLI commands</strong> — Complete CLI usage in the <a href="#/cli">CLI Reference</a></li>
        </ul>
      </section>
    </div>
  )
}
