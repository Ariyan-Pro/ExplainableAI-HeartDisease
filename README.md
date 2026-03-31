<div align="center">

<img src="logo.jpg" width="160" alt="ExplainableAI Heart Disease Predictor Logo"/>

# 🏥 ExplainableAI Heart Disease Predictor

**A clinical-grade AI system that predicts heart disease risk with 94.1% accuracy — and explains every decision in plain clinical language.**

[![Accuracy](https://img.shields.io/badge/Accuracy-94.1%25-success?style=for-the-badge)](https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor)
[![AUC Score](https://img.shields.io/badge/AUC_Score-0.967-blue?style=for-the-badge)](https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor)
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP_%2B_LIME-purple?style=for-the-badge)](https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Live Demo](https://img.shields.io/badge/🤗_Live_Demo-Hugging_Face-ff69b4?style=for-the-badge)](https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor)

[🚀 Try Live Demo](https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor) · [📖 Docs](./docs) · [🐛 Report Bug](https://github.com/Ariyan-Pro/ExplainableAI-HeartDisease/issues) · [💡 Request Feature](https://github.com/Ariyan-Pro/ExplainableAI-HeartDisease/issues)

</div>

---

## 🩺 Why This Project?

Clinical AI tools today often behave as black boxes — a model outputs a risk score, but clinicians have no way to understand *why*. This erodes trust, prevents adoption, and creates liability in high-stakes environments.

**ExplainableAI Heart Disease Predictor** bridges this gap: it combines state-of-the-art predictive performance (94.1% accuracy, 0.967 AUC) with full SHAP + LIME explainability, so every prediction comes with a human-readable clinical rationale — not just a number.

---

## 🎯 Clinical Impact

<div align="center">

| Metric | Clinical Standard | Our Performance | Improvement |
|:-------|:-----------------|:----------------|:------------|
| **Accuracy** | 85–90% | **94.1%** | **+4.1% to +9.1%** |
| **AUC Score** | 0.85–0.92 | **0.967** | **+0.047 to +0.117** |
| **Explainability** | Limited | **100% Coverage** | **Full Transparency** |
| **Federated Learning** | Rare | **85.9% FL Accuracy** | **Privacy-Preserving** |
| **Response Latency** | Varies | **< 100ms** | **Real-time Clinical Use** |

</div>

---

## ✨ Features

- **🔬 Dual Explainability (SHAP + LIME)** — Every prediction generates both global and local explanations, giving clinicians feature-level insight into risk factors driving each diagnosis.
- **📊 94.1% Predictive Accuracy** — Validated performance exceeding typical clinical decision-support benchmarks, with 0.967 ROC-AUC on held-out test data.
- **🔒 Federated Learning Support** — Train across distributed clinical sites without sharing raw patient data — GDPR/HIPAA-aligned by design (85.9% FL accuracy).
- **⚡ Production FastAPI Backend** — REST API with sub-100ms inference latency, ready for EHR system integration and clinical workflow embedding.
- **📈 Enterprise MLOps via MLflow** — Full experiment tracking, model versioning, artifact logging, and reproducible training pipelines.
- **🖥️ Interactive Gradio Dashboard** — A professional, no-code UI for clinicians to input patient data and receive annotated risk assessments in real time.
- **☁️ Cloud-Ready Deployment** — One-command deploy to Hugging Face Spaces, Streamlit Cloud, or Render — no DevOps expertise required.

---

## 📊 Performance Proof

<div align="center">

**Validation Results — 94.1% Accuracy**

![Performance Summary](./assets/performance/performance_summary.png)

</div>

---

## 🚀 Live Demo Interface

<div align="center">

**Professional Medical AI Dashboard**

[![Demo Interface](./assets/screenshots/dashboard_main.png)](https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor)

*Click the image above to launch the live demo on Hugging Face Spaces*

</div>

---

## ⚙️ Quick Start

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ariyan-Pro/ExplainableAI-HeartDisease.git
cd ExplainableAI-HeartDisease

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
python dashboard/app.py
```

The Gradio interface will be available at `http://localhost:7860`

### API Server (FastAPI)

```bash
# Start the REST API
uvicorn healthcare_model.api:app --reload --port 8000
```

API docs auto-generated at `http://localhost:8000/docs`

### Quick Prediction (Python)

```python
from healthcare_model import HeartDiseasePredictor

predictor = HeartDiseasePredictor()

patient = {
    "age": 55,
    "sex": 1,
    "cp": 2,           # Chest pain type
    "trestbps": 140,   # Resting blood pressure
    "chol": 250,       # Serum cholesterol
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,    # Max heart rate achieved
    "exang": 0,
    "oldpeak": 1.5,
    "slope": 2,
    "ca": 0,
    "thal": 2
}

result = predictor.predict(patient)
print(result["risk_score"])        # → 0.82 (high risk)
print(result["explanation"])       # → SHAP + LIME feature breakdown
```

---

## 🏗️ System Architecture

<div align="center">

**Enterprise-Grade Four-Layer Architecture**

![Architecture Diagram](./assets/architecture/error_handling_workflow.png)

</div>

The system is organized into four layers:

| Layer | Components | Role |
|:------|:-----------|:-----|
| **Interface** | Gradio Dashboard, FastAPI REST | User interaction & external integrations |
| **Explainability** | SHAP Engine, LIME Engine | Post-hoc explanation generation |
| **ML Core** | Trained Classifier, Federated Client | Prediction & privacy-preserving training |
| **MLOps** | MLflow Tracking, Artifact Store | Experiment management & reproducibility |

---

## 🔬 Explainable AI

<div align="center">

**SHAP Global Feature Importance**

![SHAP Summary](./assets/technical/shap_summary.png)

</div>

This project implements two complementary explanation strategies:

- **SHAP (SHapley Additive exPlanations)** — Provides global model interpretability by attributing each feature's contribution to predictions across the entire dataset. Clinicians can identify which biomarkers most consistently drive risk scores.

- **LIME (Local Interpretable Model-Agnostic Explanations)** — Generates patient-level, instance-specific explanations by approximating the model locally. Each prediction comes with a ranked list of contributing factors for that individual.

Together, they provide 100% prediction coverage with no unexplained outputs.

---

## 🤖 AI & Model Transparency

This project uses machine learning components. In the interest of clinical trust and EU AI Act alignment:

- **Model Type**: Supervised classification (gradient-boosted ensemble — inference from codebase)
- **Training Data**: Cleveland Heart Disease dataset (UCI ML Repository, publicly available)
- **External APIs**: None — all inference runs locally
- **Determinism**: Predictions are deterministic given fixed model weights; SHAP/LIME outputs may vary slightly with sampling configurations
- **Known Limitations**: Trained primarily on the Cleveland dataset; performance may vary on populations significantly different from the training distribution. Not a substitute for clinical diagnosis.
- **Federated Learning**: Implemented for privacy-preserving multi-site training; FL accuracy is 85.9% vs. 94.1% centralized — this trade-off is expected and disclosed.
- **User Data**: No patient data is stored, transmitted, or used for retraining in the deployed demo.

> **Disclosure**: Portions of this project's documentation were assisted by AI writing tools.

---

## 🛠️ Usage

### Running with Docker *(if applicable)*

```bash
docker build -t heart-disease-ai .
docker run -p 7860:7860 heart-disease-ai
```

### Environment Configuration

| Variable | Default | Description |
|:---------|:--------|:------------|
| `MODEL_PATH` | `healthcare_model/` | Path to trained model artifacts |
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow experiment tracking directory |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`) |
| `PORT` | `7860` | Dashboard server port |

### MLflow Experiment Tracking

```bash
# View experiment results in MLflow UI
mlflow ui --port 5000
# Navigate to http://localhost:5000
```

---

## 🤝 Contributing

Contributions are welcome from clinicians, ML engineers, and healthcare AI researchers alike. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

### Development Setup

```bash
git clone https://github.com/Ariyan-Pro/ExplainableAI-HeartDisease.git
cd ExplainableAI-HeartDisease
pip install -r requirements.txt
```

### Submitting Changes

1. Fork the repo and create your branch: `git checkout -b feat/your-feature`
2. Make your changes and add tests where applicable
3. Verify nothing is broken: `python -m pytest`
4. Open a PR — describe **what** you changed and **why**, not just what

### Areas Actively Welcoming Contributions

- Additional dataset support (MIMIC-III, Framingham)
- New explainability backends (Integrated Gradients, Captum)
- Multilingual dashboard support
- EHR integration connectors (FHIR, HL7)

---

## 🔐 Security

Please do **not** open public GitHub issues for security vulnerabilities. Instead, report them privately by emailing the repository maintainer or opening a private security advisory via GitHub's security tab.

---

## 📄 License

[MIT](LICENSE) © 2024 [Ariyan Pro](https://github.com/Ariyan-Pro)

You are free to use, modify, and distribute this software for any purpose, including commercial clinical research, with attribution.

---

## 🙏 Acknowledgments

- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) — Cleveland Heart Disease Dataset
- [SHAP](https://github.com/slundberg/shap) — Lundberg & Lee, NeurIPS 2017
- [LIME](https://github.com/marcotcr/lime) — Ribeiro et al., KDD 2016
- [MLflow](https://mlflow.org/) — Experiment tracking infrastructure
- [Gradio](https://gradio.app/) — Interactive demo interface
- [FastAPI](https://fastapi.tiangolo.com/) — Production API framework

---

## 🔗 Resources

| Resource | Link |
|:---------|:-----|
| 🤗 Live Demo | [Hugging Face Spaces](https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor) |
| 📖 Documentation | [`./docs`](./docs) |
| 🐛 Issue Tracker | [GitHub Issues](https://github.com/Ariyan-Pro/ExplainableAI-HeartDisease/issues) |
| 🤝 Contributing Guide | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

<div align="center">

Built with ❤️ for Clinical AI Transparency

*If this project helps your research or clinical work, consider starring ⭐ the repository.*

</div>
