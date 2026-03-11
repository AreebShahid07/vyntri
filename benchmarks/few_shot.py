import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from vyntri.solvers.continual import ContinualAnaCP

def benchmark():
    print("=== Vyntri vs. GD Benchmark (Synthetic Few-Shot) ===")
    
    # 1. Setup Synthetic Data
    # Simulate features from a backbone (e.g. ResNet50 -> 2048 dim)
    n_features = 512
    n_classes = 10
    n_samples = 50 # 5 shots per class
    
    print(f"Data: {n_samples} samples, {n_features} features, {n_classes} classes.")
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples)
    y_str = np.array([f"class_{i}" for i in y])
    
    # ---------------------------------------------------------
    # 2. Vyntri (Analytic Adaptation)
    # ---------------------------------------------------------
    print("\n--- Method A: Vyntri (Analytic) ---")
    start_time = time.time()
    
    solver = ContinualAnaCP(projection_dim=64)
    solver.fit(X, y_str)
    
    vyntri_time = time.time() - start_time
    print(f"Time: {vyntri_time*1000:.2f} ms")
    
    # Verify it actually learned
    acc = np.mean(solver.predict(X) == y_str)
    print(f"Train Accuracy: {acc*100:.1f}%")

    # ---------------------------------------------------------
    # 3. Standard GD (Logistic Regression in PyTorch)
    # ---------------------------------------------------------
    print("\n--- Method B: Standard Gradient Descent (PyTorch LSD) ---")
    
    # Convert to Tensor
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()
    
    # Linear Model (Classifier Head)
    model = nn.Linear(n_features, n_classes)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    # Train Loop (simulating fine-tuning head)
    epochs = 50 # Need epochs to converge
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_t)
        loss = criterion(output, y_t)
        loss.backward()
        optimizer.step()
        
    gd_time = time.time() - start_time
    print(f"Time ({epochs} epochs): {gd_time*1000:.2f} ms")
    
    # Calc Accuracy
    with torch.no_grad():
        preds = torch.argmax(model(X_t), dim=1).numpy()
    acc_gd = np.mean(preds == y)
    print(f"Train Accuracy: {acc_gd*100:.1f}%")
    
    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    speedup = gd_time / vyntri_time
    print(f"\n>>> Result: Vyntri is {speedup:.1f}x faster than simple GD Training.")

if __name__ == "__main__":
    benchmark()
