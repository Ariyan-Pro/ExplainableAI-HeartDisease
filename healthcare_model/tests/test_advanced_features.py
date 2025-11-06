# healthcare_model/tests/test_advanced_features.py
import sys
import os
import pytest

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

def test_monitoring_import():
    """Test that monitoring system can be imported"""
    try:
        from healthcare_model.monitoring import ModelMonitor, initialize_monitor
        print("✅ Monitoring import test passed")
        return True
    except ImportError as e:
        print(f"❌ Monitoring import failed: {e}")
        return False

def test_data_validation_import():
    """Test that data validation system can be imported"""
    try:
        from healthcare_model.data_validation import DataValidator, validate_incoming_data
        print("✅ Data validation import test passed")
        return True
    except ImportError as e:
        print(f"❌ Data validation import failed: {e}")
        return False

def test_error_handling_import():
    """Test that error handling system can be imported"""
    try:
        from healthcare_model.error_handling import AdvancedErrorHandler, handle_prediction_with_fallback
        print("✅ Error handling import test passed")
        return True
    except ImportError as e:
        print(f"❌ Error handling import failed: {e}")
        return False

def test_data_validation_functionality():
    """Test data validation with sample data"""
    try:
        from healthcare_model.data_validation import validate_incoming_data
        
        # Test valid data
        valid_data = {
            'age': 52, 'sex': 1, 'cp': 0, 'trestbps': 125,
            'chol': 212, 'fbs': 0, 'restecg': 1, 'thalach': 168,
            'exang': 0, 'oldpeak': 1.0, 'slope': 2, 'ca': 2, 'thal': 3
        }
        
        is_valid, errors = validate_incoming_data(valid_data)
        assert is_valid == True
        assert len(errors) == 0
        
        # Test invalid data
        invalid_data = {'age': 200}  # Age out of range
        is_valid, errors = validate_incoming_data(invalid_data)
        assert is_valid == False
        assert len(errors) > 0
        
        print("✅ Data validation functionality test passed")
        return True
    except Exception as e:
        print(f"❌ Data validation functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Advanced Features...")
    results = []
    results.append(test_monitoring_import())
    results.append(test_data_validation_import())
    results.append(test_error_handling_import())
    results.append(test_data_validation_functionality())
    
    if all(results):
        print("🎉 All advanced features tests passed!")
        exit(0)
    else:
        print("❌ Some advanced features tests failed!")
        exit(1)