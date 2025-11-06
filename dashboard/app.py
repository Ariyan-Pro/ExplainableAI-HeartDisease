# dashboard/app.py
import sys
import os
import joblib
import pandas as pd
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

# ---------- NEW: individual explanation libs ----------
import shap
import lime
import lime.lime_tabular
import base64
import io
# ----------------------------------------------------

# ---------- NEW: optional API helper ----------
def predict_via_api(patient_data):
    """Alternative prediction using API"""
    try:
        import requests
        response = requests.post(
            "http://localhost:8000/predict",
            json=patient_data,
            timeout=10
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}
# ---------------------------------------------

# ---------- NEW: explanation helpers ----------
import textwrap
def generate_global_explanations():
    """Generate and display global model explanations"""
    try:
        from explain import make_shap_summary, generate_feature_importance_plot
        from utils import load_data, split_features
        import joblib
        df = load_data()
        X_train, X_test, y_train, y_test = split_features(df)
        pipe = joblib.load(HEALTHCARE_MODEL_PATH / "pipeline_heart.joblib")
        shap_path   = make_shap_summary(X_train, pipe)
        feature_path= generate_feature_importance_plot(pipe, X_train.columns.tolist())
        return textwrap.dedent(f"""
        ✅ **Global Explanations Generated!**

        **SHAP Summary:** `{shap_path}`  
        **Feature Importance:** `{feature_path}`  

        These show what features the model considers most important overall.
        """)
    except Exception as e:
        return f"❌ Error generating explanations: {str(e)}"

def ensure_explanations_exist():
    """Auto-create explanation plots if missing"""
    shap_path   = HEALTHCARE_MODEL_PATH / "outputs" / "shap_summary.png"
    feature_path= HEALTHCARE_MODEL_PATH / "outputs" / "feature_importance.png"
    if not (shap_path.exists() and feature_path.exists()):
        print("🔄 Generating missing model explanations …")
        os.system("cd healthcare_model && python explain.py")
        print("✅ Explanations ensured.")

# ----------------------------------------------------------
#  NEW  –  individual SHAP & LIME helpers
# ----------------------------------------------------------
def generate_individual_explanation(pipe, input_data, feature_names):
    """Generate SHAP force plot for individual prediction"""
    try:
        xgb_model = pipe.named_steps['xgb']
        scaler    = pipe.named_steps['scaler']
        input_scaled = scaler.transform(input_data.reshape(1, -1))

        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_scaled)

        plt.figure(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_scaled[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()

        return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;"/>'
    except Exception as e:
        return f"❌ Explanation error: {str(e)}"

def generate_lime_explanation(pipe, input_data, feature_names, X_train):
    """Generate LIME explanation for individual prediction"""
    try:
        scaler = pipe.named_steps['scaler']
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=scaler.transform(X_train),
            feature_names=feature_names,
            mode='classification',
            random_state=42
        )

        def predict_proba_fn(x):
            return pipe.predict_proba(x)

        exp = explainer.explain_instance(
            scaler.transform(input_data.reshape(1, -1))[0],
            predict_proba_fn,
            num_features=10
        )

        fig = exp.as_pyplot_figure()
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()

        return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;"/>'
    except Exception as e:
        return f"❌ LIME explanation error: {str(e)}"
# ----------------------------------------------------------

#  NEW  –  tab content helper  (kept inside this file)
# ----------------------------------------------------------
def add_model_insights_tab():
    """Add a tab for model explanations"""
    with gr.Tab("🔍 Model Insights"):
        gr.Markdown("## How the Model Makes Decisions")

        # Load and display SHAP plot
        shap_path = HEALTHCARE_MODEL_PATH / "outputs" / "shap_summary.png"
        if shap_path.exists():
            gr.Markdown("### SHAP Feature Importance")
            gr.Image(str(shap_path), label="Global Feature Impact")

        # Load and display feature importance
        feature_path = HEALTHCARE_MODEL_PATH / "outputs" / "feature_importance.png"
        if feature_path.exists():
            gr.Markdown("### XGBoost Feature Importance")
            gr.Image(str(feature_path), label="Built-in Feature Weights")

        gr.Markdown("""
        **Understanding the Plots:**
        - **SHAP**: Shows how each feature impacts predictions (positive/negative)
        - **Feature Importance**: Shows which features the model relies on most
        """)
# ----------------------------------------------------------

# GENIUS PATH RESOLUTION - works anywhere
def get_project_root():
    """Intelligently find project root from any location"""
    current_file = Path(__file__).resolve()
    
    # Strategy 1: Look for project root from current file
    for parent in [current_file] + list(current_file.parents):
        if (parent / "healthcare_model").exists() and (parent / "dashboard").exists():
            return parent
    
    # Strategy 2: Look for common project markers
    for parent in [current_file] + list(current_file.parents):
        if (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    
    # Fallback: Assume we're in project_root/dashboard/
    return current_file.parent.parent

# Add the healthcare_model directory to Python path
PROJECT_ROOT = get_project_root()
HEALTHCARE_MODEL_PATH = PROJECT_ROOT / "healthcare_model"
sys.path.insert(0, str(HEALTHCARE_MODEL_PATH))

print(f"🔍 Project root: {PROJECT_ROOT}")
print(f"📁 Healthcare model path: {HEALTHCARE_MODEL_PATH}")

# Import from healthcare_model using genius path resolution
try:
    from utils import load_data, get_model_path
    # Use genius path resolution for model loading
    MODEL_PATH = get_model_path("pipeline_heart.joblib")
    print(f"📁 Model path: {MODEL_PATH}")
except ImportError as e:
    print(f"❌ Import error: {e}")
    # Fallback: manual path resolution
    MODEL_PATH = HEALTHCARE_MODEL_PATH / "pipeline_heart.joblib"
    print(f"🔄 Using fallback model path: {MODEL_PATH}")

# Load the trained model with robust error handling
try:
    if MODEL_PATH.exists():
        pipe = joblib.load(MODEL_PATH)
        MODEL_LOADED = True
        print("✅ Model loaded successfully!")
    else:
        MODEL_LOADED = False
        print(f"❌ Model file not found at: {MODEL_PATH}")
        print(f"📁 Available files in healthcare_model/:")
        model_dir = HEALTHCARE_MODEL_PATH
        if model_dir.exists():
            for file in model_dir.glob("*.joblib"):
                print(f"   - {file.name}")
        pipe = None
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Model loading failed: {e}")
    pipe = None

# Load data to get feature information with fallback
try:
    df = load_data()
    feature_names = df.drop(columns=['target']).columns.tolist()
    print(f"✅ Data loaded successfully: {df.shape[0]} samples")
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    # Fallback feature names
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    df = pd.DataFrame(columns=feature_names + ['target'])
    print("🔄 Using fallback feature names")

# Feature descriptions for better UX
feature_descriptions = {
    'age': 'Age in years',
    'sex': 'Sex (1 = male; 0 = female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
    'restecg': 'Resting electrocardiographic results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1 = yes; 0 = no)',
    'oldpeak': 'ST depression induced by exercise relative to rest',
    'slope': 'Slope of the peak exercise ST segment (0-2)',
    'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
    'thal': 'Thalassemia (1-3)'
}

# ----------------------------------------------------------
#  NEW  –  updated prediction function (5 outputs now)
# ----------------------------------------------------------
def predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal):
    """
    Predict heart disease probability + individual explanations
    """
    if not MODEL_LOADED:
        return "❌ Model not loaded. Please train the model first.", "", "", "", ""

    try:
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                               thalach, exang, oldpeak, slope, ca, thal]])

        probability = pipe.predict_proba(input_data)[0][1]
        prediction  = pipe.predict(input_data)[0]

        # risk level
        if probability < 0.3:
            risk_level, advice = "🟢 LOW RISK", "Maintain healthy lifestyle with regular checkups."
        elif probability < 0.7:
            risk_level, advice = "🟡 MODERATE RISK", "Consult a cardiologist for further evaluation."
        else:
            risk_level, advice = "🔴 HIGH RISK", "Seek immediate medical consultation."

        # individual explanations
        shap_html = generate_individual_explanation(pipe, input_data[0], feature_names)
        lime_html = generate_lime_explanation(pipe, input_data[0], feature_names,
                                            df.drop(columns=['target']).values)

        result_text = f"""
        ## Prediction Result

        **Heart Disease Probability:** {probability:.1%}
        **Risk Level:** {risk_level}
        **Prediction:** {'🫀 Heart Disease Detected' if prediction == 1 else '✅ No Heart Disease'}

        ### Medical Advice:
        {advice}
        """

        # risk meter plot
        fig, ax = plt.subplots(figsize=(8, 2))
        cmap = colors.LinearSegmentedColormap.from_list("risk", ["green", "yellow", "red"])
        risk_meter = ax.imshow([[probability]], cmap=cmap, aspect='auto',
                              extent=[0, 100, 0, 1], vmin=0, vmax=1)
        ax.set_xlabel('Heart Disease Risk'); ax.set_yticks([])
        ax.set_xlim(0, 100)
        ax.axvline(probability * 100, color='black', linestyle='--', linewidth=2)
        ax.text(probability * 100, 0.5, f'{probability:.1%}',
                ha='center', va='center', backgroundcolor='white', fontweight='bold')
        plt.title('Risk Assessment Meter', fontweight='bold')
        plt.tight_layout()

        return result_text, fig, "", shap_html, lime_html

    except Exception as e:
        error_msg = f"❌ Prediction error: {str(e)}"
        print(error_msg)
        return error_msg, None, "", "", ""
# ----------------------------------------------------------

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Heart Disease Predictor") as demo:
    gr.Markdown("# 🫀 Heart Disease Prediction Dashboard")
    gr.Markdown("Enter patient information to assess heart disease risk using our Explainable AI model")
    
    # Model status indicator
    status_color = "green" if MODEL_LOADED else "red"
    status_text = "✅ Model Loaded" if MODEL_LOADED else "❌ Model Not Available"
    gr.Markdown(f"### Model Status: <span style='color:{status_color}'>{status_text}</span>", 
                sanitize_html=False)
    
    if not MODEL_LOADED:
        gr.Markdown("""
        ⚠️ **Please train the model first:**
        ```bash
        cd healthcare_model
        python model.py
        ```
        """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Patient Information")
            
            # Create input components with descriptions
            inputs = []
            for feature in feature_names:
                if feature in ['age', 'trestbps', 'chol', 'thalach']:
                    # Numerical features
                    inputs.append(gr.Number(
                        label=f"{feature.upper()} - {feature_descriptions[feature]}",
                        value=df[feature].median() if not df.empty else 50
                    ))
                elif feature in ['sex', 'fbs', 'exang']:
                    # Binary features
                    inputs.append(gr.Radio(
                        label=f"{feature.upper()} - {feature_descriptions[feature]}",
                        choices=[0, 1],
                        value=0
                    ))
                else:
                    # Categorical features
                    min_val = int(df[feature].min()) if not df.empty else 0
                    max_val = int(df[feature].max()) if not df.empty else 3
                    inputs.append(gr.Slider(
                        label=f"{feature.upper()} - {feature_descriptions[feature]}",
                        minimum=min_val,
                        maximum=max_val,
                        value=min_val,
                        step=1
                    ))
        
        with gr.Column():
            gr.Markdown("### Prediction Results")
            output_text = gr.Markdown()
            output_plot = gr.Plot()
            
            # ---------- NEW: individual explanation tabs ----------
            gr.Markdown("### 🔍 Individual Prediction Explanations")
            with gr.Tab("SHAP Force Plot"):
                shap_output = gr.HTML(label="SHAP Explanation")
            with gr.Tab("LIME Explanation"):
                lime_output = gr.HTML(label="LIME Explanation")
            
            explanation_text = gr.Markdown()
    
    # Prediction button
    predict_btn = gr.Button("🔍 Predict Heart Disease Risk", variant="primary", 
                          interactive=MODEL_LOADED)
    predict_btn.click(
        fn=predict_heart_disease,
        inputs=inputs,
        outputs=[output_text, output_plot, explanation_text, shap_output, lime_output]
    )

    # ---------- NEW: Global explanation button ----------
    with gr.Row():
        explain_btn = gr.Button("🔍 Generate Global Model Insights", variant="secondary")
        explanation_output = gr.Markdown()

    explain_btn.click(
        fn=generate_global_explanations,
        inputs=[],
        outputs=[explanation_output]
    )
    # ----------------------------------------------------

    # ---------- NEW: Model Insights TAB  (inserted here) ----------
    add_model_insights_tab()
    # --------------------------------------------------------------

    # Add some examples (only if model is loaded)
    if MODEL_LOADED:
        gr.Markdown("### Example Cases")
        gr.Examples(
            examples=[
                [52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3],  # High risk
                [45, 0, 2, 130, 204, 0, 0, 172, 0, 1.4, 1, 0, 2],  # Medium risk  
                [35, 0, 1, 120, 180, 0, 0, 160, 0, 0.0, 1, 0, 1]   # Low risk
            ],
            inputs=inputs
        )

if __name__ == "__main__":
    print("\n🚀 Starting Heart Disease Prediction Dashboard...")
    print("📊 Open your browser and go to: http://127.0.0.1:7860   ")
    print("⏹️  Press Ctrl+C to stop the server")
    
    ensure_explanations_exist()   # auto-create plots on start-up
    
    try:
        demo.launch(share=False, server_port=7860, show_error=True)
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        print("💡 Try changing the port: demo.launch(server_port=7861)")