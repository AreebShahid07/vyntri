# Vyntri API Reference

## Core Library
### `vyntri.core.pipeline.VyntriPipeline`
The main entry point for adapting (training) models.

**Methods:**
- `run(dataset_path: str, solver_type="anacp", **kwargs) -> dict`
    - Analyze dataset, select best backbone, train solver.
    - `solver_type`: "anacp" (default), "fly", or "wisard".
    - `**kwargs`: `expansion_ratio` (for Fly), `num_bits` (for Wisard).
    - Returns a model context dictionary.

### `vyntri.dataset.engine.DatasetIntelligenceEngine`
Analyzes dataset difficulty before training.

**Methods:**
- `analyze(dataset_path: str) -> dict`
    - Returns comprehensive fingerprint including PVI, class balance, resolution stats.

### `vyntri.solvers.continual.ContinualAnaCP`
Analytic solver supporting incremental updates.

**Methods:**
- `fit(X, y)`: Initial training.
- `update(X_new, y_new)`: Add new data without retraining on old data.

### `vyntri.core.io`
Persistence utilities.

**Functions:**
- `save_model(model_context: dict, path: str)`: Saves trained model to `.pkl`.
- `load_model(path: str) -> dict`: Loads model ready for inference.

---

## Example Usage

```python
from vyntri.core.config import Config
from vyntri.core.pipeline import VyntriPipeline
from vyntri.core.io import save_model, load_model

# 1. Initialize Pipeline
pipeline = VyntriPipeline(Config())

# 2. Run Pipeline (Analyze -> Select -> Train)
# Supports: "anacp", "fly", "wisard"
result = pipeline.run("./data/cats_dogs", solver_type="fly")

# 3. Save Model
save_model(result, "my_model.pkl")

# 4. Load & Predict
model = load_model("my_model.pkl")
prediction = pipeline.predict_image("test.jpg", model)
print(prediction)
```
