# healthcare_model/error_handling.py
import logging
import sys
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import json

logger = logging.getLogger(__name__)

class AdvancedErrorHandler:
    """Advanced error handling with circuit breakers and fallbacks"""
    
    def __init__(self):
        self.error_counts = {}
        self.circuit_breakers = {}
        self.fallback_responses = self._setup_fallback_responses()
    
    def _setup_fallback_responses(self):
        """Setup fallback responses for different error scenarios"""
        return {
            'model_prediction': {
                'prediction': 0,
                'probability': 0.5,
                'risk_level': 'unknown',
                'confidence': 'low',
                'advice': 'System temporarily unavailable - please try again',
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'fallback': True
            },
            'data_validation': {
                'error': 'Data validation service unavailable',
                'fallback': True
            }
        }
    
    def record_error(self, error_type: str, details: str = ""):
        """Record error for circuit breaker pattern"""
        if error_type not in self.error_counts:
            self.error_counts[error_type] = []
        
        self.error_counts[error_type].append({
            'timestamp': datetime.now(),
            'details': details
        })
        
        # Clean old errors (keep last hour)
        cutoff = datetime.now().timestamp() - 3600
        self.error_counts[error_type] = [
            err for err in self.error_counts[error_type]
            if err['timestamp'].timestamp() > cutoff
        ]
        
        logger.warning(f"Error recorded: {error_type} - {details}")
    
    def is_circuit_open(self, error_type: str, threshold: int = 10, window_minutes: int = 5) -> bool:
        """Check if circuit breaker should open"""
        if error_type not in self.error_counts:
            return False
        
        # Count errors in time window
        cutoff = datetime.now().timestamp() - (window_minutes * 60)
        recent_errors = [
            err for err in self.error_counts[error_type]
            if err['timestamp'].timestamp() > cutoff
        ]
        
        if len(recent_errors) >= threshold:
            if error_type not in self.circuit_breakers:
                self.circuit_breakers[error_type] = datetime.now()
                logger.error(f"Circuit breaker opened for: {error_type}")
            return True
        
        return False
    
    def get_fallback_response(self, error_type: str, original_request: Dict = None) -> Dict:
        """Get appropriate fallback response"""
        fallback = self.fallback_responses.get(error_type, {})
        
        if original_request and 'fallback' in fallback:
            # Enhance fallback with request context
            fallback['original_request'] = {
                k: v for k, v in original_request.items() 
                if k in ['age', 'sex', 'cp']  # Include only non-sensitive fields
            }
        
        return fallback
    
    def handle_prediction_error(self, error: Exception, request_data: Dict) -> Dict:
        """Handle prediction errors with fallback"""
        error_type = 'model_prediction'
        
        # Record the error
        self.record_error(error_type, str(error))
        
        # Check circuit breaker
        if self.is_circuit_open(error_type):
            logger.error("Circuit breaker active - using fallback response")
            return self.get_fallback_response(error_type, request_data)
        
        # If circuit not open, re-raise for normal handling
        raise error
    
    def handle_validation_error(self, error: Exception, data: Dict) -> Dict:
        """Handle validation errors"""
        error_type = 'data_validation'
        self.record_error(error_type, str(error))
        
        if self.is_circuit_open(error_type):
            return self.get_fallback_response(error_type, data)
        
        # Return structured validation error
        return {
            'error': 'Data validation failed',
            'details': str(error),
            'success': False
        }

class ErrorContext:
    """Context manager for advanced error handling"""
    
    def __init__(self, operation: str, error_handler: AdvancedErrorHandler):
        self.operation = operation
        self.error_handler = error_handler
        self.start_time = datetime.now()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Error occurred - handle it
            error_details = f"{exc_type.__name__}: {str(exc_val)}"
            self.error_handler.record_error(self.operation, error_details)
            
            # Log full traceback for debugging
            logger.error(f"Error in {self.operation}: {error_details}")
            logger.debug(f"Traceback: {''.join(traceback.format_tb(exc_tb))}")
            
            # For certain operations, we might want to suppress the exception
            # and return a fallback instead
            if self.operation == 'model_prediction':
                # Don't suppress - let the API handle it
                return False
            
        return False  # Don't suppress the exception

# Global error handler instance
error_handler = AdvancedErrorHandler()

# FastAPI exception handlers
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for FastAPI"""
    error_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Log the error with context
    logger.error(
        f"Global exception handler - Error ID: {error_id}, "
        f"Path: {request.url.path}, Method: {request.method}, "
        f"Error: {str(exc)}"
    )
    
    # Determine appropriate status code
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
    else:
        status_code = 500
    
    # Record for circuit breaking
    error_handler.record_error('api_request', f"{request.url.path}: {str(exc)}")
    
    # Return structured error response
    return JSONResponse(
        status_code=status_code,
        content={
            'error_id': error_id,
            'error': 'Internal server error' if status_code == 500 else str(exc),
            'path': request.url.path,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }
    )

def handle_prediction_with_fallback(model, input_data):
    """Execute prediction with error handling and fallback"""
    with ErrorContext('model_prediction', error_handler):
        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'success': True
            }
            
        except Exception as e:
            # Let the error handler decide whether to use fallback
            return error_handler.handle_prediction_error(e, input_data)

def get_system_health():
    """Get system health including error statistics"""
    health = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'healthy',
        'error_statistics': {},
        'circuit_breakers': {}
    }
    
    # Error statistics
    for error_type, errors in error_handler.error_counts.items():
        health['error_statistics'][error_type] = {
            'total_errors': len(errors),
            'recent_errors': len([e for e in errors 
                                if (datetime.now() - e['timestamp']).total_seconds() < 300]),  # 5 minutes
            'circuit_open': error_handler.is_circuit_open(error_type)
        }
    
    # Circuit breaker status
    for cb_type, opened_at in error_handler.circuit_breakers.items():
        health['circuit_breakers'][cb_type] = {
            'opened_at': opened_at.isoformat(),
            'duration_minutes': (datetime.now() - opened_at).total_seconds() / 60
        }
    
    # Determine overall status
    open_circuits = sum(1 for stats in health['error_statistics'].values() 
                       if stats.get('circuit_open', False))
    
    if open_circuits > 0:
        health['overall_status'] = 'degraded'
    elif any(stats['recent_errors'] > 5 for stats in health['error_statistics'].values()):
        health['overall_status'] = 'unstable'
    
    return health

if __name__ == "__main__":
    # Test the error handling system
    health = get_system_health()
    print("System Health:", json.dumps(health, indent=2))