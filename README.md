# Vyntri

**Training-less vision intelligence for edge devices.**

Vyntri lets low-power devices learn new objects in **milliseconds**. It replaces GPU-heavy training loops with closed-form linear algebra, enabling one-shot and few-shot learning directly on Raspberry Pi, Jetson, and consumer CPUs.

## What Vyntri Is (and Isn't)

**Vyntri is for:**
- Image classification on **CPU / edge devices** (RPi, Jetson, laptops)
- **Folder-per-class** datasets (one subfolder per category, images inside)
- Situations where you need to learn from **1-50 images per class** in under a second
- Adding new classes **without retraining** (continual learning with zero forgetting)

**Vyntri is NOT:**
- A general-purpose deep learning training framework
- A replacement for PyTorch/TensorFlow on large-scale GPU workloads
- Designed for tasks beyond image classification (no detection, segmentation, etc.)

## How It Works

```
Your images (folder-per-class)
       |
       v
 Dataset Intelligence -- analyzes image stats, class balance, complexity
       |
       v
 Backbone Selection ---- picks MobileNetV3 (fast) or ResNet18 (robust)
       |
       v
 Feature Extraction ---- runs images through the frozen backbone
       |
       v
 Analytic Adaptation --- solves for classifier weights via linear algebra
       |                  (no gradient descent, no epochs, no GPUs)
       v
 Ready to predict
```

**Training speed:** ~50ms on CPU for 100 images (3600x faster than fine-tuning).
**Inference speed:** Same as the backbone alone (~45ms on RPi 4).

## Installation

```bash
pip install vyntri
```

For the webcam demo (optional):
```bash
pip install vyntri[webcam]
```

## Quick Start -- Python API

```python
from vyntri.core import VyntriPipeline, Config, save_model, load_model

# Adapt a model to your dataset in one call
pipeline = VyntriPipeline(Config())
result = pipeline.run("./my_dataset", solver_type="anacp")

# Save and reuse
save_model(result, "model.pkl")

# Later: load and predict
ctx = load_model("model.pkl")
label = ctx["solver"].predict(features)
```

### Continual Learning (add new classes without retraining)

```python
from vyntri.solvers import ContinualAnaCP

solver = ContinualAnaCP(projection_dim=128)
solver.fit(features_batch1, labels_batch1)

# Later, new class appears:
solver.update(features_batch2, labels_batch2)
# Old classes are preserved exactly -- zero forgetting
```

## Quick Start -- CLI

```bash
# Analyze a dataset
vyntri analyze ./my_dataset

# Train (adapt) a model
vyntri train ./my_dataset --save_path model.pkl --solver anacp

# Predict a single image
vyntri predict ./photo.jpg model.pkl

# Add new classes to an existing model
vyntri update ./new_data model.pkl
```

## Dataset Format

Vyntri expects a **folder-per-class** layout:

```
my_dataset/
  cat/
    img1.jpg
    img2.jpg
  dog/
    img1.jpg
    img2.jpg
  bird/
    img1.jpg      <-- even 1 image per class works
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

## Solvers

| Solver | Best For | Speed |
|---|---|---|
| `anacp` (default) | General use, best accuracy | ~50ms adapt |
| `fly` | Noisy data, bio-inspired sparse coding | ~30ms adapt |
| `wisard` | Ultra-fast inference on simple patterns | ~10ms adapt |

## Webcam Demo

Teach your webcam to recognize objects in real-time:

```bash
python examples/demo_webcam.py
```

Press **Space** to capture and label an object. It's recognized instantly on the next frame.

## Documentation

- [User Guide](docs/user_guide.md) -- CLI and API details
- [Concepts](docs/concepts.md) -- How analytic adaptation works
- [Edge Deployment](docs/edge_deployment.md) -- Running on RPi / Jetson
- [API Reference](docs/api_reference.md) -- Full API docs

## License

MIT
