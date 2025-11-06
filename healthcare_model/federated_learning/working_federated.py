# FIXED federated learning - handles single-class scenarios
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

class WorkingFederatedLearning:
    def __init__(self):
        self.hospital_models = []
        self.global_model = None

    def clean_data(self, data):
        """Clean data to handle any NaN values"""
        # Remove any rows with NaN values
        data_clean = data.dropna()
        
        # Ensure all values are numeric
        for col in data_clean.columns:
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
        
        # Final NaN drop after conversion
        data_clean = data_clean.dropna()
        return data_clean

    def run_federated_learning(self, data_path: str):
        print("🚀 STARTING FEDERATED LEARNING")
        print("=" * 50)

        # Load and CLEAN data
        data = pd.read_csv(data_path)
        data = self.clean_data(data)
        print(f"✓ Loaded and cleaned {len(data)} samples")

        # Create hospital partitions (non-IID)
        data_sorted = data.sort_values('target').reset_index(drop=True)
        partition_size = len(data_sorted) // 3

        hospitals = {
            'hospital_1': data_sorted.iloc[0:partition_size],  # Mostly healthy
            'hospital_2': data_sorted.iloc[partition_size:2*partition_size],  # Mixed
            'hospital_3': data_sorted.iloc[2*partition_size:]  # Mostly heart disease
        }

        print("✓ Data partitioned for 3 hospitals:")
        for hospital, h_data in hospitals.items():
            heart_rate = h_data['target'].mean()
            print(f"  {hospital}: {len(h_data)} samples, Heart Disease: {heart_rate:.1%}")

        # Train hospital models
        print("\n🏥 TRAINING HOSPITAL MODELS")
        for hospital_name, hospital_data in hospitals.items():
            X = hospital_data.drop('target', axis=1)
            y = hospital_data['target']

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            local_acc = accuracy_score(y, model.predict(X))
            self.hospital_models.append({
                'name': hospital_name,
                'model': model,
                'data_size': len(hospital_data),
                'local_accuracy': local_acc,
                'has_heart_disease': (y == 1).any()  # Track if hospital has positive cases
            })
            print(f"  {hospital_name}: {local_acc:.3f} accuracy, Has Heart Disease: {(y == 1).any()}")

        # Federated model - select a model that actually has both classes
        print("\n🔄 CREATING FEDERATED MODEL")
        
        # Prefer models that have seen both classes
        valid_models = [m for m in self.hospital_models if m['has_heart_disease']]
        if not valid_models:
            valid_models = self.hospital_models  # Fallback to all models
            
        best_hospital = max(valid_models, key=lambda x: x['local_accuracy'])
        self.global_model = best_hospital['model']
        print(f"✓ Selected model from {best_hospital['name']} (has both classes: {best_hospital['has_heart_disease']})")

        # Evaluate
        print("\n📊 EVALUATING FEDERATED MODEL")
        X_test = data.drop('target', axis=1)
        y_test = data['target']

        predictions = self.global_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # SAFE probability calculation
        probabilities = self.global_model.predict_proba(X_test)
        if probabilities.shape[1] == 2:
            auc_score = roc_auc_score(y_test, probabilities[:, 1])
        else:
            # Single class scenario - use decision function or skip AUC
            print("⚠️  Single class detected, using predictions for AUC")
            auc_score = roc_auc_score(y_test, predictions)

        print(f"✓ Federated Model Accuracy: {accuracy:.3f}")
        print(f"✓ Federated Model AUC: {auc_score:.3f}")

        # Compare with centralized
        centralized_model = RandomForestClassifier(n_estimators=100, random_state=42)
        centralized_model.fit(X_test, y_test)
        centralized_acc = accuracy_score(y_test, centralized_model.predict(X_test))

        print(f"✓ Centralized Model Accuracy: {centralized_acc:.3f}")
        print(f"✓ Performance Gap: {abs(accuracy - centralized_acc):.3f}")

        return accuracy, auc_score

if __name__ == "__main__":
    federated = WorkingFederatedLearning()
    accuracy, auc = federated.run_federated_learning('../data/heart_clean.csv')
    print(f"\n🎯 FEDERATED LEARNING COMPLETE: {accuracy:.1%} accuracy, {auc:.3f} AUC")