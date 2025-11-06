# 🏗️ System Architecture Overview

## Explainable AI for Heart Disease Prediction

### System Components
The system consists of 5 modular layers designed for clinical interpretability and production reliability.

### Architecture Layers

1. **Data Layer** — Medical data preprocessing (UCI Heart Disease dataset)
2. **Model Layer** — XGBoost pipeline with Scikit-learn preprocessing
3. **Explainability Layer** — SHAP (global) + LIME (local) interpretability
4. **Interface Layer** — Gradio dashboard + FastAPI REST endpoints
5. **Evaluation Layer** — Comprehensive metrics + MLflow tracking

### Technical Performance
- **Accuracy**: 94.1% (optimized XGBoost)
- **Federated Learning**: 85.9% accuracy across 3 hospitals
- **Explainability**: 100% prediction coverage
- **Response Time**: <100ms per prediction

\\\mermaid
graph TD
    A[📊 Data Layer<br/>UCI Heart Dataset] --> B[🤖 Model Layer<br/>XGBoost Pipeline]
    B --> C[🔍 Explainability Layer<br/>SHAP + LIME]
    C --> D[💻 Interface Layer<br/>Gradio + FastAPI]
    D --> E[📈 Evaluation Layer<br/>MLflow + Metrics]
    
    F[🏥 Federated Learning] --> B
    G[⚡ Real-time Monitoring] --> E
    H[🔒 Security Validation] --> D
\\\

### Deployment Architecture
- **Local**: Direct Python deployment with port 7860/8000
- **Cloud**: Multi-platform compatible (Hugging Face, Streamlit, Render)
- **MLOps**: Automated retraining with Optuna optimization
