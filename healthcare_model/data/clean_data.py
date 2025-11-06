import pandas as pd
import numpy as np

# Load the processed Cleveland data
df = pd.read_csv('processed.cleveland.data', header=None)

# Define proper column names (from heart-disease.names)
column_names = [
    'age',      # Age in years
    'sex',      # Sex (1 = male; 0 = female)
    'cp',       # Chest pain type (1-4)
    'trestbps', # Resting blood pressure
    'chol',     # Serum cholesterol in mg/dl
    'fbs',      # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    'restecg',  # Resting electrocardiographic results
    'thalach',  # Maximum heart rate achieved
    'exang',    # Exercise induced angina (1 = yes; 0 = no)
    'oldpeak',  # ST depression induced by exercise
    'slope',    # Slope of peak exercise ST segment
    'ca',       # Number of major vessels colored by fluoroscopy
    'thal',     # Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
    'target'    # Diagnosis of heart disease (0 = no disease; 1-4 = disease present)
]

df.columns = column_names

# Replace '?' with NaN
df = df.replace('?', np.nan)

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert target to binary classification (0 = no disease, 1 = disease)
df['target'] = (df['target'] > 0).astype(int)

# Display dataset info
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nTarget Distribution:")
print(df['target'].value_counts())
print("\nFirst 5 rows:")
print(df.head())

# Save as clean CSV
df.to_csv('heart_clean.csv', index=False)
print(f"\n✅ Clean dataset saved as 'heart_clean.csv'")
print(f"✅ Total records: {len(df)}")
print(f"✅ Features: {len(df.columns) - 1}")
print(f"✅ Healthy (0): {len(df[df['target'] == 0])}")
print(f"✅ Heart Disease (1): {len(df[df['target'] == 1])}")
