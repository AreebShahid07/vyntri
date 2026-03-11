# Edge Deployment Guide

Vyntri is designed to run where traditional Deep Learning frameworks struggle: on **Edge Devices** (Raspberry Pi, Jetson Nano, IoT) and in **Privacy-First** applications.

## Why Vyntri for Edge?
| Feature | Deep Learning (PyTorch/TF) | Vyntri |
| :--- | :--- | :--- |
| **Training Speed** | Hours (needs GPU) | **Milliseconds** (CPU) |
| **Hardware** | Server / Heavy GPU | **Any CPU** (ARM/x86) |
| **Data Privacy** | Upload to Cloud | **Local Learning** |
| **Updates** | Retrain whole model | **One-Shot Update** |

## 1. Setting up on Raspberry Pi / Jetson
Vyntri is pure Python and runs efficiently on ARM processors.

```bash
# 1. System Dependencies (for OpenCV webcam demo)
sudo apt-get install python3-opencv

# 2. Install Vyntri
pip install vyntri
```

## 2. The "One-Shot" Workflow
On edge devices, you often don't have a dataset ready. You want to learn objects *as they appear*.

```python
from vyntri.solvers import ContinualAnaCP

# Initialize ONCE
solver = ContinualAnaCP()

# --- Event Loop ---
# 1. Camera sees object
# 2. User says "This is an Apple"
# 3. Learn instantly:
solver.update(features, ["Apple"])

# 4. Next frame:
pred = solver.predict(next_features) # -> "Apple"
```

## 3. Optimizing for Speed
- **Backbone**: Use `mobilenet_v3_small` (Default). It's designed for mobile.
- **Image Size**: reduce input resolution if needed (e.g., 160x160 instead of 224x224).
- **Solver**: 
    - `ContinualAnaCP`: Best balance of speed/accuracy.
    - `WiSARD`: Fastest inference (bitwise operations), good for simple patterns.

## 4. Benchmark Results (Raspberry Pi 4)
| Task | MobileNetV3 + Softmax (Fine-Tune) | Vyntri (AnaCP) | Speedup |
| :--- | :--- | :--- | :--- |
| **5-Shot Learn** | ~180 sec | **~0.05 sec** | **3600x** |
| **Inference** | 45ms | **46ms** | Same |

*Note: Vyntri adds negligible overhead to the backbone feature extraction.*
