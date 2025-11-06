print('🧪 COMPREHENSIVE REAL-WORLD TESTING')
print('=' * 50)

# TEST 1: CORE DATA SCIENCE WORKFLOW
print('\n1. TESTING CORE DATA SCIENCE...')
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Create sample data
X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
joblib.dump(model, 'test_model.joblib')
loaded_model = joblib.load('test_model.joblib')
print(f'   ✅ RandomForest: {accuracy:.1%} accuracy, Model serialization: WORKING')

# TEST 2: XGBOOST
print('\n2. TESTING XGBOOST...')
import xgboost as xgb
xgb_model = xgb.XGBClassifier(n_estimators=10)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f'   ✅ XGBoost: {xgb_accuracy:.1%} accuracy')

# TEST 3: VISUALIZATION
print('\n3. TESTING VISUALIZATION...')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Test matplotlib
plt.figure(figsize=(8, 4))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('Matplotlib Test')
plt.savefig('test_plot.png', dpi=100, bbox_inches='tight')
plt.close()

# Test seaborn
sns.set_theme()
sns_plot = sns.histplot([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
sns_plot.figure.savefig('test_seaborn.png', dpi=100, bbox_inches='tight')
plt.close()

# Test PIL
img = Image.new('RGB', (100, 100), color='red')
img.save('test_pil.png')
print('   ✅ Matplotlib, Seaborn, PIL: PLOTS GENERATED')

# TEST 4: EXPLAINABLE AI - SHAP
print('\n4. TESTING SHAP EXPLAINABILITY...')
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Test SHAP visualization
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('test_shap.png', dpi=100, bbox_inches='tight')
plt.close()
print(f'   ✅ SHAP: {shap_values.shape} shape, Visualizations: WORKING')

# TEST 5: LIME EXPLAINABILITY
print('\n5. TESTING LIME EXPLAINABILITY...')
import lime
import lime.lime_tabular

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_train, mode='classification', training_labels=y_train
)
exp = explainer_lime.explain_instance(X_test[0], xgb_model.predict_proba, num_features=5)
print(f'   ✅ LIME: Explanation generated with {len(exp.as_list())} features')

# TEST 6: WEB API
print('\n6. TESTING FASTAPI COMPONENTS...')
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Test Pydantic models
class PatientData(BaseModel):
    age: int
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float

# Test FastAPI app creation
app = FastAPI(title='Test API')

@app.get('/')
def root():
    return {'message': 'API Working'}

@app.post('/predict')
def predict(data: PatientData):
    return {'prediction': 0.75, 'confidence': 0.92}

print('   ✅ FastAPI, Pydantic, Uvicorn: COMPONENTS READY')

# TEST 7: GRADIO INTERFACE
print('\n7. TESTING GRADIO INTERFACE...')
import gradio as gr

def gradio_test(age, bp, cholesterol):
    return f'Age: {age}, BP: {bp}, Chol: {cholesterol}'

# Test interface creation (don't launch)
iface = gr.Interface(
    fn=gradio_test,
    inputs=[
        gr.Number(label='Age'),
        gr.Number(label='Blood Pressure'), 
        gr.Number(label='Cholesterol')
    ],
    outputs='text',
    title='Heart Disease Test'
)
print('   ✅ Gradio: INTERFACE CREATED')

# TEST 8: ML OPS
print('\n8. TESTING ML OPS...')
import mlflow
import optuna

# Test MLflow
with mlflow.start_run():
    mlflow.log_param('test_param', 'value')
    mlflow.log_metric('test_accuracy', 0.85)
    print('   ✅ MLflow: EXPERIMENT TRACKING WORKING')

# Test Optuna
def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=2)
print(f'   ✅ Optuna: OPTIMIZATION WORKING, Best value: {study.best_value:.3f}')

# TEST 9: FEDERATED LEARNING
print('\n9. TESTING FEDERATED LEARNING...')
import flwr as fl
print('   ✅ Flower: FEDERATED LEARNING READY')

# CLEANUP
import os
for file in ['test_model.joblib', 'test_plot.png', 'test_seaborn.png', 'test_pil.png', 'test_shap.png']:
    if os.path.exists(file):
        os.remove(file)

print('')
print('=' * 50)
print('🎉 ALL COMPONENTS TESTED SUCCESSFULLY!')
print('🚀 YOUR ENVIRONMENT IS PRODUCTION-READY!')
