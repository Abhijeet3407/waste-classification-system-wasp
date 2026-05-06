"""
Dataset Download Helper
Downloads and prepares the TrashNet dataset
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

import config

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_trashnet():
    """Download and extract TrashNet dataset"""
    print("="*60)
    print("TrashNet Dataset Downloader")
    print("="*60)
    
    # TrashNet GitHub repository
    repo_url = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"
    download_path = "trashnet.zip"
    extract_path = "trashnet_temp"
    
    print("\n📥 Downloading TrashNet dataset...")
    print(f"Source: {repo_url}")
    
    try:
        # Download with progress bar
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(repo_url, download_path, reporthook=t.update_to)
        
        print("✓ Download complete!\n")
        
        # Extract
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print("✓ Extraction complete!\n")
        
        # Move to correct location
        print("📁 Organizing dataset...")
        
        # Find the dataset folder (should be trashnet-master/data)
        source_data = Path(extract_path) / "trashnet-master" / "data"
        
        if source_data.exists():
            # Copy class folders to data/raw/
            for class_folder in source_data.iterdir():
                if class_folder.is_dir():
                    dest_folder = config.RAW_DATA_DIR / class_folder.name
                    
                    # Remove if exists
                    if dest_folder.exists():
                        shutil.rmtree(dest_folder)
                    
                    # Copy
                    shutil.copytree(class_folder, dest_folder)
                    
                    num_images = len(list(dest_folder.glob("*.jpg")))
                    print(f"  ✓ {class_folder.name}: {num_images} images")
            
            print("\n✓ Dataset organized successfully!")
        else:
            print("✗ Error: Dataset structure not as expected")
            return False
        
        # Cleanup
        print("\n🧹 Cleaning up...")
        os.remove(download_path)
        shutil.rmtree(extract_path)
        print("✓ Cleanup complete!\n")
        
        # Print summary
        print("="*60)
        print("DATASET READY!")
        print("="*60)
        print(f"Location: {config.RAW_DATA_DIR}")
        print("\nNext steps:")
        print("1. Run data preparation: python utils/data_loader.py")
        print("2. Train model: python train/train.py")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        
        # Cleanup on error
        if os.path.exists(download_path):
            os.remove(download_path)
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
        
        return False


def download_kaggle_dataset():
    """Instructions for downloading from Kaggle"""
    print("="*60)
    print("Kaggle Waste Dataset Download")
    print("="*60)
    print("\nTo download datasets from Kaggle:")
    print("\n1. Install Kaggle CLI:")
    print("   pip install kaggle")
    print("\n2. Get Kaggle API credentials:")
    print("   - Go to https://www.kaggle.com/account")
    print("   - Click 'Create New API Token'")
    print("   - Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
    print("\n3. Download dataset:")
    print("   kaggle datasets download -d techsash/waste-classification-data")
    print("\n4. Extract to data/raw/ and organize by class folders")
    print("\nPopular Kaggle waste datasets:")
    print("  - techsash/waste-classification-data")
    print("  - mostafaabla/garbage-classification")
    print("  - asdasdasasdas/garbage-classification")
    print("="*60)


def check_existing_data():
    """Check if data already exists"""
    if not config.RAW_DATA_DIR.exists():
        return False
    
    class_folders = [d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()]
    
    if len(class_folders) == 0:
        return False
    
    print("="*60)
    print("Existing Dataset Found")
    print("="*60)
    print(f"\nLocation: {config.RAW_DATA_DIR}")
    print(f"Classes found: {len(class_folders)}\n")
    
    total_images = 0
    for folder in class_folders:
        num_images = len(list(folder.glob("*.*")))
        total_images += num_images
        print(f"  - {folder.name}: {num_images} images")
    
    print(f"\nTotal images: {total_images}")
    print("="*60)
    
    return True


def main():
    """Main function"""
    print("\n🗑️  Waste Classification Dataset Helper\n")
    
    # Check if data already exists
    if check_existing_data():
        response = input("\nDataset already exists. Download anyway? (y/N): ")
        if response.lower() != 'y':
            print("Using existing dataset.")
            return
    
    # Show options
    print("\nDataset Options:")
    print("1. Download TrashNet (Recommended - 2527 images, 6 classes)")
    print("2. Instructions for Kaggle datasets")
    print("3. Use custom dataset (manual setup)")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        success = download_trashnet()
        if success:
            print("\n✓ Ready to proceed with data preparation!")
            response = input("Run data preparation now? (Y/n): ")
            if response.lower() != 'n':
                import subprocess
                subprocess.run([sys.executable, "utils/data_loader.py"])
    
    elif choice == '2':
        download_kaggle_dataset()
    
    elif choice == '3':
        print("\n" + "="*60)
        print("Custom Dataset Setup")
        print("="*60)
        print("\n1. Create folder structure:")
        print(f"   {config.RAW_DATA_DIR}/")
        print("   ├── class1/")
        print("   │   ├── image1.jpg")
        print("   │   └── image2.jpg")
        print("   ├── class2/")
        print("   │   └── ...")
        print("\n2. Update config.py with your class names:")
        print("   CLASS_NAMES = ['class1', 'class2', ...]")
        print("\n3. Run data preparation:")
        print("   python utils/data_loader.py")
        print("="*60)
    
    elif choice == '4':
        print("Goodbye!")
    
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
