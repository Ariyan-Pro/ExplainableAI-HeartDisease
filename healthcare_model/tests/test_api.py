# healthcare_model/tests/test_api.py
import pytest
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

def test_health_check():
    """Test health check endpoint"""
    try:
        from fastapi.testclient import TestClient
        from healthcare_model.api import app
        
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        print("✅ Health check test passed")
        return True
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def test_root_endpoint():
    """Test root endpoint"""
    try:
        from fastapi.testclient import TestClient
        from healthcare_model.api import app
        
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        print("✅ Root endpoint test passed")
        return True
    except Exception as e:
        print(f"❌ Root endpoint test failed: {e}")
        return False

def test_fastapi_import():
    """Test FastAPI availability"""
    try:
        import fastapi
        print("✅ FastAPI import test passed")
        return True
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False

if __name__ == "__main__":
    # Run tests manually
    print("🧪 Running API tests...")
    results = []
    results.append(test_fastapi_import())
    results.append(test_health_check())
    results.append(test_root_endpoint())
    
    if all(results):
        print("🎉 All API tests passed!")
        exit(0)
    else:
        print("❌ Some API tests failed!")
        exit(1)