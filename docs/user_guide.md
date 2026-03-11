# Vyntri User Guide

## Installation
Vyntri is a pure Python library.

```bash
pip install vyntri
```

**Requirements**:
- Python >= 3.9
- `torch`, `torchvision`, `numpy`, `pillow`, `scikit-learn`

---

## CLI Usage
Vyntri provides a CLI for all core tasks.

### 1. Analyze a Dataset
Get insights into your data before modeling.
```bash
vyntri analyze ./my_dataset
```
**Output**: stats about image counts, class balance, and complexity scores.

### 2. Train a Model (Adaptation)
Adapt a vision model to your dataset in seconds.
```bash
vyntri train ./my_dataset --save_path model.pkl --solver anacp
```
- `--solver`:
    - `anacp`: (Default) Analytic Contrastive Projection. Best for general use.
    - `fly`: Bio-inspired sparse coding. Good for noisy data.
    - `wisard`: Weightless neural network. Extremely fast inference.

### 3. Make Predictions
Classify a single image using a saved model.
```bash
vyntri predict ./test_image.jpg model.pkl
```

### 4. Continual Learning (Update)
Add new classes or samples to an existing model **without retraining from scratch**.
```bash
vyntri update ./new_data model.pkl
```
- The `model.pkl` is updated in-place.
- The model now recognizes classes from both the old and new datasets.

---

## Python API Usage

### Basic Workflow
```python
from vyntri.core import Config, VyntriPipeline, save_model

# Initialize
pipeline = VyntriPipeline(Config())

# Run Pipeline
# This analyzes, selects backbone, extracts features, and adapts the solver.
result = pipeline.run("./data/train", solver_type="anacp")

# Save
save_model(result, "my_model.pkl")
```
