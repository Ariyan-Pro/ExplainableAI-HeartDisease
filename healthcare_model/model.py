# healthcare_model/model.py
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from utils import load_data, split_features, get_model_path, get_output_path

# GENIUS PATH RESOLUTION - works anywhere
MODEL_PATH = get_model_path("xgb_heart_model.joblib")
PIPE_PATH = get_model_path("pipeline_heart.joblib")

def train_and_save():
    print("🚀 Starting model training...")
    print(f"📁 Model will be saved to: {PIPE_PATH}")
    
    df = load_data()
    X_train, X_test, y_train, y_test = split_features(df)
    
    print(f"📊 Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"📊 Test data: {X_test.shape[0]} samples")

    # simple pipeline: scale + xgboost
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
    ])

    print("🔄 Training model...")
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:,1]
    
    print("\n📈 Model Performance:")
    print("=" * 40)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Save both pipeline and standalone model
    joblib.dump(pipe, PIPE_PATH)
    joblib.dump(pipe.named_steps['xgb'], MODEL_PATH)
    
    print(f"\n✅ Saved pipeline to {PIPE_PATH}")
    print(f"✅ Saved model to {MODEL_PATH}")
    print(f"🎉 Training completed successfully!")
    
    return pipe, X_test, y_test

if __name__ == "__main__":
    try:
        train_and_save()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise