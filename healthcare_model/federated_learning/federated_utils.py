"""
Utility functions for Federated Learning implementation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPartitioner:
    """Partition data for different hospitals in federated learning"""
    
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self.hospital_data = {}
    
    def partition_by_hospital(self, n_hospitals: int = 3, 
                            partition_strategy: str = "iid") -> Dict:
        """
        Partition data for multiple hospitals
        
        Args:
            n_hospitals: Number of hospitals to partition for
            partition_strategy: "iid" (uniform) or "non-iid" (skewed)
            
        Returns:
            Dictionary of hospital data partitions
        """
        if partition_strategy == "iid":
            return self._iid_partition(n_hospitals)
        elif partition_strategy == "non-iid":
            return self._non_iid_partition(n_hospitals)
        else:
            raise ValueError("Invalid partition strategy")
    
    def _iid_partition(self, n_hospitals: int) -> Dict:
        """Independent and identically distributed partitioning"""
        hospital_data = {}
        data_copy = self.data.copy()
        
        # Shuffle data
        data_copy = data_copy.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into equal parts
        partition_size = len(data_copy) // n_hospitals
        
        for i in range(n_hospitals):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < n_hospitals - 1 else len(data_copy)
            
            hospital_data[f"hospital_{i+1}"] = data_copy.iloc[start_idx:end_idx]
            logger.info(f"Hospital {i+1} data size: {len(hospital_data[f'hospital_{i+1}'])}")
        
        return hospital_data
    
    def _non_iid_partition(self, n_hospitals: int) -> Dict:
        """Non-IID partitioning to simulate real-world data skew"""
        hospital_data = {}
        data_copy = self.data.copy()
        
        # Sort by target to create label skew
        data_copy = data_copy.sort_values('target')
        
        # Create skewed partitions
        total_samples = len(data_copy)
        samples_per_hospital = total_samples // n_hospitals
        
        for i in range(n_hospitals):
            start_idx = i * samples_per_hospital
            end_idx = start_idx + samples_per_hospital if i < n_hospitals - 1 else total_samples
            
            hospital_data[f"hospital_{i+1}"] = data_copy.iloc[start_idx:end_idx]
            
            # Calculate label distribution
            label_dist = hospital_data[f"hospital_{i+1}"]['target'].value_counts(normalize=True)
            logger.info(f"Hospital {i+1}: {len(hospital_data[f'hospital_{i+1}'])} samples, "
                       f"Label distribution: {label_dist.to_dict()}")
        
        return hospital_data

def save_hospital_data(hospital_data: Dict, base_path: str):
    """Save partitioned data for each hospital"""
    for hospital_name, data in hospital_data.items():
        file_path = f"{base_path}/{hospital_name}_data.csv"
        data.to_csv(file_path, index=False)
        logger.info(f"Saved {hospital_name} data to {file_path}")

def load_hospital_data(hospital_name: str, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load hospital data and split into features and target"""
    data = pd.read_csv(data_path)
    X = data.drop('target', axis=1)
    y = data['target']
    return X, y

class FederationMetrics:
    """Track and analyze federated learning metrics"""
    
    def __init__(self):
        self.round_metrics = []
        self.hospital_contributions = {}
    
    def add_round_metrics(self, round_num: int, metrics: Dict):
        """Add metrics for a federation round"""
        metrics['round'] = round_num
        self.round_metrics.append(metrics)
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of federation performance"""
        return pd.DataFrame(self.round_metrics)
    
    def plot_convergence(self):
        """Plot convergence of federated learning"""
        import matplotlib.pyplot as plt
        
        if not self.round_metrics:
            logger.warning("No metrics to plot")
            return
        
        df = self.get_performance_summary()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['round'], df.get('accuracy', []), marker='o', label='Accuracy')
        plt.plot(df['round'], df.get('auc_score', []), marker='s', label='AUC Score')
        
        plt.xlabel('Federation Round')
        plt.ylabel('Performance')
        plt.title('Federated Learning Convergence')
        plt.legend()
        plt.grid(True)
        plt.show()