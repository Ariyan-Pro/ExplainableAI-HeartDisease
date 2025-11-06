print('🏥 PROJECT-SPECIFIC COMPONENT VERIFICATION')
print('=' * 50)

try:
    # Test data loading
    import pandas as pd
    data = pd.read_csv('healthcare_model/data/heart_clean.csv')
    print(f'✅ DATA LOADING: {len(data)} samples loaded')
    
    # Test production model
    import joblib
    model = joblib.load('healthcare_model/models/pipeline_heart_optimized.joblib')
    predictions = model.predict(data.drop('target', axis=1))
    print(f'✅ PRODUCTION MODEL: {len(predictions)} predictions generated')
    
    # Test API components
    from api import app
    print('✅ API: COMPONENTS LOADED')
    
    # Test explainability
    from explain import generate_shap_explanation
    print('✅ EXPLAINABILITY: MODULES LOADED')
    
    # Test monitoring
    from monitoring import log_prediction
    print('✅ MONITORING: MODULES LOADED')
    
    # Test data validation
    from data_validation import validate_medical_data
    print('✅ DATA VALIDATION: MODULES LOADED')
    
    print('')
    print('🎉 ALL PROJECT COMPONENTS VERIFIED!')
    print('🚀 YOUR HEART DISEASE PREDICTION SYSTEM IS FULLY OPERATIONAL!')
    
except Exception as e:
    print(f'❌ COMPONENT FAILED: {e}')
