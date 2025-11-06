XAI-Cardio: Clinical-Grade Explainable AI for Heart Disease Prediction
Clinical-Grade AI Transparency | 94.1% Accuracy | Production-Ready
https://www.python.org/
https://github.com/Ariyan-Pro/XAI-Cardio
LICENSE
https://www.mlflow.org/
https://flower.tech/
Executive Summary
XAI-Cardio is a production-ready clinical AI system that delivers unparalleled accuracy and transparency for heart disease prediction. Built to address modern healthcare challenges, this system bridges the gap between cutting-edge machine learning and medical ethics. Key highlights include:
A 94.1% model accuracy, exceeding the industry standard (85–90%).
100% explainability coverage through SHAP and LIME frameworks.
A 85.9% federated learning benchmark, showcasing privacy-preserving innovation.
Sub-100ms response times for real-time clinical-grade performance.
This project evolves across three phases, advancing from a foundational prototype to a sophisticated enterprise-grade solution with advanced MLOps and security protocols.
Key Metrics & Competitive Edge
Table
Copy
Metric	Baseline (Phase 1)	Optimized (Phase 3)	Competitive Position
Model Accuracy	85%	94.1%	+4.6% to +9.1% improvement over industry benchmarks (85–90%).
ROC-AUC	0.891	0.967	Excellent discrimination capability.
Response Time	<1 second	<100ms (P95)	Clinical-grade speed, ~5x faster than a 500ms benchmark.
Federated Learning	Not Started	85.9%	Privacy-preserving model training across decentralized datasets using Flower 1.8.0.
System Reliability	Basic Metrics	99.9% Uptime Target	Enterprise-grade reliability with structured logging, error handling, and circuit breakers.
Architecture Overview
Mermaid
Copy
Code
Preview
graph TD
    A[Patient Data] --> B[XGBoost Model]
    B --> C[SHAP Analysis]
    B --> D[LIME Explanations]
    C --> E[Clinical Dashboard]
    D --> E
    F[Federated Learning] --> B
    G[MLflow Tracking] --> E
The system employs a modular, microservices-ready architecture optimized for clinical environments:
XGBoost Model: Achieves 94.1% accuracy with Optuna-driven hyperparameter optimization.
Explainability: Dual framework (SHAP + LIME) for global and local interpretability.
Federated Learning: Privacy-preserving training across simulated hospitals (Flower 1.8.0).
MLOps Pipeline: MLflow 2.8.1 for experiment tracking and model versioning.
Production API: FastAPI + Uvicorn for scalable prediction endpoints.
Clinical Dashboard: Gradio interface for real-time model interpretation.
Quick Start
bash
Copy
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch clinical dashboard
cd dashboard && python app.py
# Access dashboard at: http://localhost:7860

# 3. Start API server
uvicorn healthcare_model.api:app --host 0.0.0.0 --port 8000
# API Documentation: http://localhost:8000/docs
Advanced Features & Differentiators
Security & Reliability
Medical Input Validation: Pydantic ensures clinically valid inputs (e.g., age 1–120 years, blood pressure 50–200 mmHg).
Error Handling: Enterprise-grade resilience with structured logging and circuit breakers.
Dependency Stability: 42 optimized dependencies after resolving 15+ conflicts.
Research Innovation
Explainable AI: SHAP (v0.49.1) for global interpretability and LIME for instance-level explanations.
Federated Learning: Demonstrates decentralized training with 85.9% accuracy using Flower.
Multi-Modal Readiness: Integrates 10 ECG signal features alongside 13 clinical features.
Documentation
Comprehensive guides for clinical teams, developers, and researchers:
Architecture Diagrams: System components and workflow.
Clinical Validation: Performance benchmarks and testing results.
Deployment Guides: Instructions for cloud platforms (AWS, GCP, Azure) and Hugging Face Spaces.
Research Papers: Detailing methodology and regulatory considerations.
Live Deployment
Hugging Face Spaces: View on Hugging Face
Streamlit Cloud: Deploy on Streamlit
Citation
bibtex
Copy
@software{XAI_Cardio_2025,
  author = {Ariyan-Pro},
  title = {XAI-Cardio: Clinical-Grade Explainable AI for Heart Disease Prediction},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/Ariyan-Pro/XAI-Cardio}
}
Clinical Disclaimer
This system is designed for research and educational purposes. Always consult healthcare professionals for medical decisions.
Built with ❤️ for transparent medical AI