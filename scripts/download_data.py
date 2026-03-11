import os
import requests
import zipfile
import shutil

DATASET_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
ZIP_PATH = "cats_and_dogs.zip"
EXTRACT_DIR = "dataset_temp"
FINAL_DIR = "dataset"

def download_file(url, filename):
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")

def setup_dataset():
    if os.path.exists(FINAL_DIR):
        print(f"'{FINAL_DIR}' already exists. Skipping download.")
        return

    download_file(DATASET_URL, ZIP_PATH)

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

    # Structure is PetImages/Cat and PetImages/Dog
    base_dir = os.path.join(EXTRACT_DIR, "PetImages")
    
    if os.path.exists(FINAL_DIR):
        shutil.rmtree(FINAL_DIR)
        
    print(f"Moving files to {FINAL_DIR}...")
    shutil.move(base_dir, FINAL_DIR)

    # Cleanup
    print("Cleaning up...")
    os.remove(ZIP_PATH)
    shutil.rmtree(EXTRACT_DIR)
    
    # Clean up corrupted images (known issue in this dataset)
    # Some files are empty or not jpg
    print("Sanitizing dataset...")
    import glob
    for msg in glob.glob(os.path.join(FINAL_DIR, "*", "*.jpg")):
        if os.path.getsize(msg) == 0:
            os.remove(msg)
    
    print(f"Dataset ready at: {os.path.abspath(FINAL_DIR)}")
    print("You can now use this path in the Vyntri CLI.")

if __name__ == "__main__":
    setup_dataset()
