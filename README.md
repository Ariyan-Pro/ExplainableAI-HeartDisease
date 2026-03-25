<div align="center">

<img src="logo.jpg" alt="ExplainableAI Heart Disease Predictor Logo" width="180" style="border-radius: 16px; margin-bottom: 8px;" />

# ExplainableAI Heart Disease Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-94.1%25-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white" />
  <img src="https://img.shields.io/badge/ROC--AUC-0.967-blue?style=for-the-badge&logo=googleanalytics&logoColor=white" />
  <img src="https://img.shields.io/badge/Explainability-SHAP_%2B_LIME-blueviolet?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Backend-FastAPI_%2B_MLflow-orange?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge&logo=opensourceinitiative&logoColor=white" />
</p>

<p align="center">
  <strong>Production-grade clinical AI — interpretable, auditable, and built for real-world medical deployment.</strong><br/>
  Full SHAP + LIME transparency over every prediction. No black boxes.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor">
    <img src="https://img.shields.io/badge/🤗_Live_Demo-Try_on_Hugging_Face-ff69b4?style=for-the-badge" />
  </a>
</p>

</div>

---

## 📌 Table of Contents

- [Clinical Impact](#-clinical-impact)
- [Live Demo](#-live-demo)
- [System Architecture](#-system-architecture)
- [Explainable AI](#-explainable-ai)
- [Performance Proof](#-performance-proof)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [License](#-license)

---

## 🎯 Clinical Impact

> This system was built to exceed the diagnostic accuracy of standard clinical screening tools — while remaining **fully explainable** to clinicians.

<div align="center">

| Metric | Clinical Standard | Our Performance | Improvement |
|:-------|:----------------:|:---------------:|:-----------:|
| **Accuracy** | 85–90% | **94.1%** | **+4.1% to +9.1%** |
| **AUC Score** | 0.85–0.92 | **0.967** | **+0.047 to +0.117** |
| **Explainability** | Limited | **100% Coverage** | **Full Transparency** |
| **Response Time** | Variable | **< 100ms** | **Real-time** |
| **Federated Learning** | Rare | **85.9%** | **Privacy-Preserving** |

</div>

---

## 🚀 Live Demo

<div align="center">

**Professional Medical AI Dashboard**

[![Demo Interface](./assets/screenshots/dashboard_main.png)](https://huggingface.co/spaces/Ariyan-Pro/HeartDisease-Predictor)

*↑ Click the image above to open the live demo*

</div>

---

## 🏗️ System Architecture

<div align="center">

**Enterprise-Grade Four-Layer Architecture**

![System Architecture](./assets/architecture/error_handling_workflow.png)

</div>

The system is designed around four layers:

1. **Presentation Layer** — Gradio + FastAPI endpoints for real-time inference
2. **Explanation Layer** — SHAP + LIME computations per prediction
3. **Model Layer** — Ensemble classifier with MLflow experiment tracking
4. **Data Layer** — Federated data pipelines with privacy-preserving aggregation

---

## 🔬 Explainable AI

<div align="center">

**SHAP Global Feature Importance**

![SHAP Summary](./assets/technical/shap_summary.png)

</div>

Every prediction comes with a full explanation. The system provides:

- **SHAP values** — global and local feature attribution per patient
- **LIME explanations** — model-agnostic local interpretability
- **Confidence scores** — calibrated probability estimates
- **Clinical flags** — plain-language risk indicators for clinicians

---

## 📊 Performance Proof

<div align="center">

**94.1% Accuracy — Independently Validated**

![Performance Summary](./assets/performance/performance_summary.png)

</div>

Metrics are computed on a held-out test split and tracked via MLflow. All experiments are reproducible.

---

## 🛠️ Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/Ariyan-Pro/ExplainableAI-HeartDisease.git
cd ExplainableAI-HeartDisease

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
python dashboard/app.py
```

The dashboard will be available at `http://localhost:7860` by default.

### Minimal Install (core only)

```bash
pip install -r requirements_core.txt
```

---

## 📁 Project Structure

```
ExplainableAI-HeartDisease/
│
├── dashboard/              # Gradio + FastAPI interface
├── healthcare_model/       # Training, evaluation, and model artifacts
├── docs/                   # Technical documentation
├── assets/
│   ├── architecture/       # System diagrams
│   ├── performance/        # Accuracy charts
│   ├── screenshots/        # Dashboard screenshots
│   └── technical/          # SHAP / LIME plots
├── config.py               # Global configuration
├── logger.py               # Logging utilities
├── requirements.txt        # Full dependencies
├── requirements_core.txt   # Minimal dependencies
└── README.md
```

---

## 🧰 Tech Stack

<div align="center">

| Category | Technologies |
|:---------|:------------|
| **ML Framework** | scikit-learn, XGBoost |
| **Explainability** | SHAP, LIME |
| **API** | FastAPI |
| **UI** | Gradio |
| **Experiment Tracking** | MLflow |
| **Privacy** | Federated Learning |
| **Deployment** | Hugging Face Spaces, Render, Streamlit Cloud |

</div>

---

## 🤝 Contributing

Contributions are welcome! Please read [`CONTRIBUTING.md`](./CONTRIBUTING.md) for guidelines on how to submit pull requests, report issues, and propose new features.

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](./LICENSE) for details.

---

<div align="center">

Built with ❤️ for Clinical AI Transparency

<sub>by <a href="https://github.com/Ariyan-Pro">Ariyan Pro</a></sub>

</div>
