# healthcare_model/utils.py
import pandas as pd
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

class PathMaster:
    """Genius-level path resolution that works anywhere, forever"""
    
    def __init__(self):
        self._project_root = self._find_project_root()
        self._ensure_paths()
    
    def _find_project_root(self):
        """Intelligently find project root using multiple fallback strategies"""
        # Strategy 1: Look for project markers
        possible_roots = [
            Path(__file__).parent.parent,  # healthcare_model/../
            Path.cwd(),                    # Current directory
            self._find_by_markers(),       # Look for project markers
        ]
        
        for root in possible_roots:
            if self._is_project_root(root):
                return root
        
        # Final fallback: current file location
        return Path(__file__).parent.parent
    
    def _find_by_markers(self):
        """Look for project markers (.git, requirements.txt, etc.)"""
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / ".git").exists() or (parent / "requirements.txt").exists():
                return parent
        return current
    
    def _is_project_root(self, path):
        """Check if path contains our project structure"""
        required = [
            path / "healthcare_model",
            path / "healthcare_model" / "data",
            path / "healthcare_model" / "utils.py"
        ]
        return all(item.exists() for item in required)
    
    def _ensure_paths(self):
        """Ensure all critical paths exist"""
        critical_paths = [
            self.get("healthcare_model/data"),
            self.get("healthcare_model/models")
        ]
        for path in critical_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
    
    def get(self, relative_path):
        """Get absolute path for any relative path"""
        return self._project_root / relative_path
    
    def resolve_data_path(self, fallback_path="healthcare_model/data/heart_clean.csv"):
        """Smart data path resolution with multiple fallbacks"""
        possible_locations = [
            self.get(fallback_path),
            self.get("data/heart_clean.csv"),
            Path(__file__).parent / "data" / "heart_clean.csv",
        ]
        
        for location in possible_locations:
            if location.exists():
                print(f"🎯 Found data at: {location}")
                return location
        
        # If no file found, show helpful error
        available_files = list(self.get("healthcare_model/data").glob("*.csv"))
        raise FileNotFoundError(
            f"❌ Data file not found! Tried: {[str(p) for p in possible_locations]}\n"
            f"📁 Available files: {[f.name for f in available_files]}"
        )

# Global instance - this is the genius part
_path_master = PathMaster()

def load_data(path=None):
    """Ultra-robust data loading that works from anywhere"""
    if path is None:
        data_path = _path_master.resolve_data_path()
    else:
        data_path = _path_master.get(path)
    
    print(f"📂 Loading data from: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    original_shape = df.shape
    df = df.drop_duplicates().dropna()
    final_shape = df.shape
    
    if original_shape != final_shape:
        print(f"🧹 Cleaned data: {original_shape[0]} → {final_shape[0]} rows")
    
    print(f"✅ Successfully loaded: {final_shape[0]} rows, {final_shape[1]} columns")
    return df

def split_features(df, target_col='target', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_model_path(filename):
    """Get absolute path for model files"""
    return _path_master.get(f"healthcare_model/{filename}")

def get_output_path(filename):
    """Get absolute path for output files"""
    output_dir = _path_master.get("healthcare_model/outputs")
    output_dir.mkdir(exist_ok=True)
    return output_dir / filename