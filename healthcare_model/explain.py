# healthcare_model/explain.py
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data, split_features, get_model_path, get_output_path

# Try to import SHAP and LIME with proper error handling
try:
    import shap
    # Force SHAP to use compatible numpy functions
    shap.utils._safe_isinstance = lambda x, y: isinstance(x, y)
    SHAP_AVAILABLE = True
except ImportError as e:
    SHAP_AVAILABLE = False
    print(f"SHAP not available: {e}")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError as e:
    LIME_AVAILABLE = False
    print(f"LIME not available: {e}")

# GENIUS PATH RESOLUTION - works anywhere
PIPE_PATH = get_model_path("pipeline_heart.joblib")
MODEL_PATH = get_model_path("best_heart_model.joblib")
SHAP_IMAGE_PATH = get_output_path("shap_summary.png")
FEATURE_IMPORTANCE_PATH = get_output_path("feature_importance.png")

def make_shap_summary(X_train, model_pipeline, save_path=SHAP_IMAGE_PATH):
    if not SHAP_AVAILABLE:
        print("SHAP not installed - skipping SHAP summary")
        return None
        
    try:
        print("Generating SHAP summary...")
        
        # Extract model and scaler from pipeline
        xgb = model_pipeline.named_steps['xgb']
        scaler = model_pipeline.named_steps['scaler']
        
        # Transform data
        X_scaled = scaler.transform(X_train)
        
        # Use TreeExplainer for XGBoost (more efficient)
        explainer = shap.TreeExplainer(xgb)
        
        # Calculate SHAP values - use a subset for speed
        sample_size = min(100, len(X_scaled))
        X_sample = X_scaled[:sample_size]
        shap_values = explainer.shap_values(X_sample)
        
        # Create the summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=X_train.columns, show=False)
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP summary saved to {save_path}")
        
        # Also print top features
        mean_abs_shap = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        print("\nTop features by SHAP importance:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
            
        return save_path
        
    except Exception as e:
        print(f"❌ SHAP error: {e}")
        print("But don't worry - we still have LIME and feature importance!")
        return None

def explain_instance_with_lime(X_train_df, model_pipeline, instance, num_features=6):
    if not LIME_AVAILABLE:
        print("LIME not installed - skipping LIME explanation")
        return []
        
    try:
        scaler = model_pipeline.named_steps['scaler']
        xgb = model_pipeline.named_steps['xgb']

        X_train = X_train_df.values
        explainer = LimeTabularExplainer(X_train,
                                         feature_names=X_train_df.columns,
                                         class_names=['NoDisease','Disease'],
                                         mode='classification')
        
        def predict_proba_fn(x):
            x_scaled = scaler.transform(x)
            return xgb.predict_proba(x_scaled)
            
        exp = explainer.explain_instance(instance.values, predict_proba_fn, num_features=num_features)
        return exp.as_list()
        
    except Exception as e:
        print(f"LIME error: {e}")
        return []

def generate_feature_importance_plot(model_pipeline, feature_names, save_path=FEATURE_IMPORTANCE_PATH):
    """Backup: Generate feature importance using XGBoost's built-in method"""
    xgb = model_pipeline.named_steps['xgb']
    importances = xgb.feature_importances_
    
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("XGBoost Built-in Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color='lightblue', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path

if __name__ == "__main__":
    print("="*60)
    print("STEP 4: GENERATING MODEL EXPLANATIONS")
    print("="*60)
    
    # 🎯 GENIUS PATH RESOLUTION IN ACTION
    print(f"📁 Pipeline path: {PIPE_PATH}")
    print(f"📁 Model path: {MODEL_PATH}")
    
    try:
        df = load_data()
        X_train, X_test, y_train, y_test = split_features(df)
        pipe = joblib.load(PIPE_PATH)
        
        # 1. SHAP Summary (Global Explainability)
        if SHAP_AVAILABLE:
            shap_result = make_shap_summary(X_train, pipe)
        else:
            print("\n💡 Install SHAP for global explanations: pip install shap==0.44.0")
        
        # 2. LIME Explanation (Local Explainability)  
        if LIME_AVAILABLE:
            print("\n" + "="*40)
            print("LIME LOCAL EXPLANATION")
            print("="*40)
            lime_explanation = explain_instance_with_lime(X_train, pipe, X_test.iloc[0])
            print("Features influencing this specific prediction:")
            print("(Negative = reduces risk, Positive = increases risk)")
            for feature, importance in lime_explanation:
                risk = "🔻 reduces risk" if importance < 0 else "🔺 increases risk"
                print(f"  {feature}: {importance:.4f} ({risk})")
        else:
            print("\n💡 LIME not available for local explanations")
        
        # 3. Backup: Built-in feature importance
        print("\n" + "="*40)
        print("BUILT-IN FEATURE IMPORTANCE")
        print("="*40)
        generate_feature_importance_plot(pipe, X_train.columns.tolist())
        print("✓ Feature importance plot saved as 'feature_importance.png'")
        
        print("\n" + "🎉" * 20)
        print("STEP 4 COMPLETED!")
        print("You now have multiple layers of model explainability!")
        print("Ready for STEP 5: Interactive Dashboard!")
        print("🎉" * 20)
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("1. Check if data files exist in healthcare_model/data/")
        print("2. Run from project root or healthcare_model/ directory")
        print("3. Ensure pipeline_heart.joblib exists")
        raise