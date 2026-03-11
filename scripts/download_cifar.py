import os
import shutil
import torchvision
from PIL import Image
from tqdm import tqdm

DATASET_DIR = "dataset"

def setup_cifar10():
    if os.path.exists(DATASET_DIR):
        print(f"Cleaning existing {DATASET_DIR}...")
        shutil.rmtree(DATASET_DIR)
    
    os.makedirs(DATASET_DIR)
    
    print("Downloading CIFAR-10...")
    # Download to temp
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    
    print("Extracting images to folder structure...")
    # Class names
    classes = dataset.classes
    
    for cls in classes:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)
        
    # Save images
    count = 0
    # Limit to 100 images per class for speed if needed, or full dataset (50k)
    # Full dataset is fine, it's small (32x32).
    
    for idx, (img, label) in enumerate(tqdm(dataset)):
        cls_name = classes[label]
        img.save(os.path.join(DATASET_DIR, cls_name, f"{idx}.jpg"))
        count += 1
        
    print(f"Saved {count} images to {DATASET_DIR}")
    
    # Cleanup downloaded tar
    if os.path.exists("./data"):
        shutil.rmtree("./data")

if __name__ == "__main__":
    setup_cifar10()
