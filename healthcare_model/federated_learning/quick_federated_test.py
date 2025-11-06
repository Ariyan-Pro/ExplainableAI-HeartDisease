"""
Quick test of federated learning setup
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

def simulate_federated_learning():
    """Simulate federated learning without actual network communication"""
    print("=== SIMULATING FEDERATED LEARNING ===")
    
    # Load and partition data
    data = pd.read_csv('../data/heart_clean.csv')
    
    # Create hospital partitions (non-IID)
    hospital_data = {}
    data_sorted = data.sort_values('target')
    
    partitions = [
        data_sorted.iloc[0:100],      # Hospital 1: Mostly healthy
        data_sorted.iloc[100:200],    # Hospital 2: Mixed
        data_sorted.iloc[200:297]     # Hospital 3: Mostly heart disease
    ]
    
    hospital_models = []
    hospital_performance = []
    
    # Train local models
    for i, hospital_data in enumerate(partitions):
        print(f"\n--- Hospital {i+1} Local Training ---")
        print(f"Samples: {len(hospital_data)}, Heart Disease Rate: {hospital_data['target'].mean():.2f}")
        
        X_local = hospital_data.drop('target', axis=1)
        y_local = hospital_data['target']
        
        # Train local model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_local, y_local)
        hospital_models.append(model)
        
        # Local performance
        local_pred = model.predict(X_local)
        local_acc = accuracy_score(y_local, local_pred)
        print(f"Local Accuracy: {local_acc:.4f}")
    
    # Federated aggregation (simple averaging of predictions)
    print(f"\n=== FEDERATED AGGREGATION ===")
    
    # Test on global test set
    X_global = data.drop('target', axis=1)
    y_global = data['target']
    
    # Get predictions from all hospitals
    all_predictions = []
    for i, model in enumerate(hospital_models):
        pred_proba = model.predict_proba(X_global)[:, 1]
        all_predictions.append(pred_proba)
        print(f"Hospital {i+1} Global AUC: {roc_auc_score(y_global, pred_proba):.4f}")
    
    # Average predictions (federated aggregation)
    federated_predictions = np.mean(all_predictions, axis=0)
    federated_auc = roc_auc_score(y_global, federated_predictions)
    
    print(f"\n=== RESULTS ===")
    print(f"Federated Model AUC: {federated_auc:.4f}")
    
    # Compare with centralized model
    centralized_model = RandomForestClassifier(n_estimators=50, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_global, y_global, test_size=0.2, random_state=42)
    centralized_model.fit(X_train, y_train)
    centralized_pred = centralized_model.predict_proba(X_test)[:, 1]
    centralized_auc = roc_auc_score(y_test, centralized_pred)
    
    print(f"Centralized Model AUC: {centralized_auc:.4f}")
    print(f"Performance Gap: {abs(federated_auc - centralized_auc):.4f}")

if __name__ == "__main__":
    simulate_federated_learning()