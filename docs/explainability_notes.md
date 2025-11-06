# 🔍 Explainability Layer Documentation

## Dual-Approach Interpretability System

### SHAP (SHapley Additive exPlanations)
**Purpose**: Global model interpretability across entire dataset
**Implementation**: TreeExplainer for XGBoost models
**Key Insights**:
- Top features: \ca\ (number of major vessels), \	hal\ (thalassemia), \cp\ (chest pain type)
- Global feature importance rankings
- Expected value: -0.243 (base prediction)

**Visualizations**:
- Summary plots for feature importance
- Force plots for individual predictions
- Dependence plots for feature relationships

### LIME (Local Interpretable Model-agnostic Explanations)
**Purpose**: Local interpretability for individual predictions
**Implementation**: TabularExplainer for per-instance explanations
**Key Benefits**:
- Case-by-case medical reasoning
- Trust building with clinicians
- Regulatory compliance support

### Clinical Validation
- **Coverage**: 100% of predictions explained
- **Latency**: <200ms per explanation
- **Accuracy**: Local surrogate model fidelity >92%
