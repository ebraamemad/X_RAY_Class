# dataset.py
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from pathlib import Path

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, transform=None, image_cache=None):
        self.dataframe = dataframe.copy() #سخة من البيانات
        self.transform = transform
        self.image_cache = image_cache or {}
        
        # Build image cache if not provided
        if not self.image_cache:
            self._build_image_cache()
        
        # Fix paths in dataframe
        self._fix_dataframe_paths()
    
    def _build_image_cache(self):
        """Build a cache of filename -> full_path mappings"""
        print("Building image cache...")
        image_extensions = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
        
        # Search for all image files recursively
        for root, dirs, files in os.walk('.'):
            # Skip hidden directories and common non-image directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'mlruns', '.git']]
            
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file)
                    # Store both the filename and the full path
                    self.image_cache[file] = full_path
        
        print(f"Found {len(self.image_cache)} images in cache")
    
    def _fix_dataframe_paths(self):
        """Fix the paths in the dataframe using the image cache"""
        corrected_paths = []
        found_count = 0
        
        for idx, row in self.dataframe.iterrows():
            original_path = row['resized']
            filename = os.path.basename(original_path)
            
            # Check if file exists at original path
            if os.path.exists(original_path):
                corrected_paths.append(original_path)
                found_count += 1
            # Check if file exists in cache
            elif filename in self.image_cache:
                corrected_paths.append(self.image_cache[filename])
                found_count += 1
            else:
                # Keep original path (will cause error if used)
                corrected_paths.append(original_path)
                if idx < 5:  # Only print first 5 missing files
                    print(f"Warning: Could not find {filename}")
        
        self.dataframe['resized'] = corrected_paths
        print(f"Fixed {found_count} out of {len(self.dataframe)} image paths")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['resized']
        
        try:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            # Load image
            image = Image.open(img_path).convert('RGB')
            label = 1 if self.dataframe.iloc[idx]['clas'] == 'PNEUMONIA' else 0
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

# Global image cache to share between datasets
_global_image_cache = None

def get_global_image_cache():
    """Get or create global image cache"""
    global _global_image_cache
    if _global_image_cache is None:
        _global_image_cache = {}
        print("Creating global image cache...")
        image_extensions = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
        
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'mlruns', '.git']]
            
            for file in files:
                if any(file.endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(root, file).replace('\\', '/')
                    _global_image_cache[file] = full_path
        
        print(f"Global cache built with {len(_global_image_cache)} images")
    
    return _global_image_cache