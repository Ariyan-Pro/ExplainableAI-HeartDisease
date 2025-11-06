# healthcare_model/train_with_mlflow.py
import mlflow
import mlflow.sklearn
import joblib
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use absolute import
from healthcare_model.utils import load_data, split_features

def train_with_tracking(use_optimized_params=True):
    """Train model with MLflow experiment tracking"""
    
    # Set up MLflow
    mlflow.set_experiment("Heart_Disease_Prediction")
    
    with mlflow.start_run():
        # Load data
        df = load_data()
        X_train, X_test, y_train, y_test = split_features(df)
        
        # Use optimized parameters from your previous run
        if use_optimized_params:
            params = {
                'n_estimators': 100,
                'max_depth': 8,
                'learning_rate': 0.13189353462617695,
                'subsample': 0.6007131041878475,
                'colsample_bytree': 0.9919604509578513,
                'reg_alpha': 0.2780055569191314,
                'reg_lambda': 4.792495635496788,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            run_name = "Optimized_XGBoost"
        else:
            params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            run_name = "Baseline_XGBoost"
        
        mlflow.set_tag("mlflow.runName", run_name)
        
        # Log parameters
        mlflow.log_params(params)
        
        # Create and train pipeline
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(**params))
        ])
        
        pipe.fit(X_train, y_train)
        
        # Predictions and metrics
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:,1]
        
        accuracy = accuracy_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)
        
        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "roc_auc": roc_auc
        })
        
        # Log model
        mlflow.sklearn.log_model(pipe, "model")
        
        # Generate and log SHAP plot
        try:
            xgb_model = pipe.named_steps['xgb']
            scaler = pipe.named_steps['scaler']
            X_scaled = scaler.transform(X_train)
            
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_scaled[:100])  # Sample for speed
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_scaled[:100], feature_names=X_train.columns, show=False)
            plt.tight_layout()
            plt.savefig("shap_summary_mlflow.png")
            mlflow.log_artifact("shap_summary_mlflow.png")
            plt.close()
            print("✅ SHAP plot generated and logged!")
        except Exception as e:
            print(f"SHAP visualization failed: {e}")
        
        print(f"✅ Experiment logged! Accuracy: {accuracy:.3f}, ROC-AUC: {roc_auc:.3f}")
        
        return pipe

if __name__ == "__main__":
    train_with_tracking(use_optimized_params=True)