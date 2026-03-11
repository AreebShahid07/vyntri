# Vyntri API Reference

## Core Library
### `vyntri.core.pipeline.VyntriPipeline`
The main entry point for running the entire adaptation pipeline.

**Methods:**
- `run(dataset_path: str, solver_type="anacp", **kwargs) -> dict`
    - Full pipeline execution: Analyze -> Select -> Extract -> Adapt.
    - `solver_type`: "anacp" (default), "fly", or "wisard".
    - `**kwargs`: Solver-specific parameters (e.g., `expansion_ratio` for Fly).
    - Returns a model context dictionary containing the trained solver and metadata.

### `vyntri.dataset.engine.DatasetIntelligenceEngine`
Analyzes dataset characteristics *before* training.

**Methods:**
- `analyze(dataset_path: str) -> dict`
    - Computes fingerprints: class balance, image stats, PVI (Projection Variability Index).
    - Returns a comprehensive dictionary of dataset metadata.

### `vyntri.core.io`
Persistence utilities for saving/loading models.

**Functions:**
- `save_model(model_context: dict, path: str)`: Saves the trained model context to a `.pkl` file.
- `load_model(path: str) -> dict`: Loads a model context ready for inference or updates.

---

## Solvers

### `vyntri.solvers.continual.ContinualAnaCP`
**Analytic Contrastive Projection** with continual learning capabilities.

- **`fit(X, y)`**: Initial adaptation. Solves for weights analytically.
- **`update(X_new, y_new)`**: Incrementally updates the model with new data using Recursive Least Squares. No retraining on old data required.
- **`predict(X)`**: Returns class predictions.

### `vyntri.solvers.fly.FlyCL`
**Bio-Inspired Sparse Coding** based on the fruit fly olfactory circuit.

- **`fit(X, y)`**: Learns sparse representations by projecting up to a high-dimensional space (expansion).
- **`predict(X)`**: Sparse similarity matching.

### `vyntri.solvers.wisard.WiSARD`
**Weightless Neural Network** (RAM-based).

- **`fit(X, y)`**: Fast, memory-efficient learning.
- **`predict(X)`**: Extremely fast binary pattern matching.

---

## Backbones
Vyntri supports pre-trained backbones for feature extraction. The `ModelSelector` automatically chooses the best one based on dataset fingerprints, or you can manually override.

- **`mobilenet_v3_small`**: Lightweight, fast. Good for mobile/edge. (Default for simple tasks)
- **`resnet18`**: Robust, standard. Good general-purpose feature extractor.
