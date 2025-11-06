# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Project structure
    PROJECT_ROOT = Path(__file__).parent
    MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH", "healthcare_model/pipeline_heart_optimized.joblib")
    DATA_PATH = PROJECT_ROOT / os.getenv("DATA_PATH", "healthcare_model/data/heart_clean.csv")
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Security & Logging
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    MAX_PREDICTION_AGE_DAYS = int(os.getenv("MAX_PREDICTION_AGE_DAYS", "30"))
    
    # Feature flags
    ENABLE_SHAP = os.getenv("ENABLE_SHAP", "True").lower() == "true"
    ENABLE_LIME = os.getenv("ENABLE_LIME", "True").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate critical configuration"""
        if not cls.DATA_PATH.exists():
            raise FileNotFoundError(f"Data file not found: {cls.DATA_PATH}")
        
        if not cls.MODEL_PATH.exists():
            # Try fallback model
            fallback_model = cls.PROJECT_ROOT / "healthcare_model/pipeline_heart.joblib"
            if not fallback_model.exists():
                raise FileNotFoundError(f"Model file not found: {cls.MODEL_PATH} or {fallback_model}")
            else:
                cls.MODEL_PATH = fallback_model
                print(f"⚠️ Using fallback model: {cls.MODEL_PATH}")
        
        print("✅ Configuration validated successfully")
        return True

settings = Settings()