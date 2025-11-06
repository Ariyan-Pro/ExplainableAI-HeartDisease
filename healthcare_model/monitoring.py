# healthcare_model/monitoring.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)

class ModelMonitor:
    """Advanced model performance monitoring and drift detection"""
    
    def __init__(self, model_path, data_path, monitoring_window=30):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.monitoring_window = monitoring_window
        self.metrics_history = self._load_metrics_history()
        
    def _load_metrics_history(self):
        """Load historical metrics from file"""
        # FIXED: Create monitoring directory properly
        monitoring_dir = Path('healthcare_model/monitoring')
        monitoring_dir.mkdir(parents=True, exist_ok=True)  # This line fixes it
        
        history_file = monitoring_dir / 'metrics_history.json'
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        return []
    
    def _save_metrics_history(self):
        """Save metrics history to file"""
        history_file = Path('healthcare_model/monitoring/metrics_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def calculate_model_metrics(self, X_test, y_test, model):
        """Calculate comprehensive model performance metrics"""
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(2 * (precision_score(y_test, y_pred, zero_division=0) * 
                                     recall_score(y_test, y_pred, zero_division=0)) / 
                                     (precision_score(y_test, y_pred, zero_division=0) + 
                                      recall_score(y_test, y_pred, zero_division=0) + 1e-8)),
                'data_size': len(X_test),
                'positive_rate': float(y_test.mean())
            }
            return metrics
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return None
    
    def detect_performance_drift(self, current_metrics, threshold=0.05):
        """Detect significant performance degradation"""
        if len(self.metrics_history) < 2:
            return False, "Insufficient historical data"
        
        # Get recent metrics (last monitoring_window days)
        recent_cutoff = datetime.now() - timedelta(days=self.monitoring_window)
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) > recent_cutoff
        ]
        
        if not recent_metrics:
            return False, "No recent metrics for comparison"
        
        # Calculate baseline performance
        baseline_roc_auc = np.mean([m['roc_auc'] for m in recent_metrics])
        current_roc_auc = current_metrics['roc_auc']
        
        performance_drop = baseline_roc_auc - current_roc_auc
        drift_detected = performance_drop > threshold
        
        alert_msg = ""
        if drift_detected:
            alert_msg = f"Performance drift detected: ROC-AUC dropped by {performance_drop:.3f}"
            logger.warning(alert_msg)
        
        return drift_detected, alert_msg
    
    def check_data_drift(self, current_data, reference_data=None):
        """Simple data drift detection using summary statistics"""
        if reference_data is None:
            # Use training data as reference
            from utils import load_data
            reference_data = load_data().drop(columns=['target'])
        
        drift_metrics = {}
        
        for column in current_data.columns:
            if column in reference_data.columns:
                # Compare basic statistics
                current_mean = current_data[column].mean()
                reference_mean = reference_data[column].mean()
                current_std = current_data[column].std()
                reference_std = reference_data[column].std()
                
                # Simple drift detection (z-score based)
                mean_drift = abs(current_mean - reference_mean) / (reference_std + 1e-8)
                std_drift = abs(current_std - reference_std) / (reference_std + 1e-8)
                
                drift_metrics[column] = {
                    'mean_drift': float(mean_drift),
                    'std_drift': float(std_drift),
                    'drift_detected': mean_drift > 2.0 or std_drift > 2.0  # 2 sigma threshold
                }
        
        return drift_metrics
    
    def monitor_model_health(self, X_test, y_test, model):
        """Comprehensive model health monitoring"""
        # Calculate current metrics
        current_metrics = self.calculate_model_metrics(X_test, y_test, model)
        if not current_metrics:
            return {"error": "Failed to calculate metrics"}
        
        # Detect performance drift
        performance_drift, drift_message = self.detect_performance_drift(current_metrics)
        
        # Detect data drift
        data_drift = self.check_data_drift(X_test)
        
        # Update history
        self.metrics_history.append(current_metrics)
        self._save_metrics_history()
        
        # Generate health report
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'current_performance': current_metrics,
            'performance_drift': {
                'detected': performance_drift,
                'message': drift_message,
                'threshold_exceeded': performance_drift
            },
            'data_drift': data_drift,
            'model_age_days': self.get_model_age(),
            'health_status': 'healthy' if not performance_drift else 'degrading'
        }
        
        logger.info(f"Model health check: {health_report['health_status']}")
        return health_report
    
    def get_model_age(self):
        """Calculate model age in days"""
        model_mtime = datetime.fromtimestamp(self.model_path.stat().st_mtime)
        return (datetime.now() - model_mtime).days
    
    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        if not self.metrics_history:
            return {"error": "No monitoring data available"}
        
        latest_metrics = self.metrics_history[-1]
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'model_performance': latest_metrics,
            'trend_analysis': self.analyze_performance_trend(),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def analyze_performance_trend(self):
        """Analyze performance trends over time"""
        if len(self.metrics_history) < 3:
            return "Insufficient data for trend analysis"
        
        recent_metrics = self.metrics_history[-5:]  # Last 5 measurements
        
        roc_trend = np.array([m['roc_auc'] for m in recent_metrics])
        trend_slope = np.polyfit(range(len(roc_trend)), roc_trend, 1)[0]
        
        if trend_slope > 0.01:
            return "Improving trend"
        elif trend_slope < -0.01:
            return "Declining trend - investigate"
        else:
            return "Stable performance"
    
    def generate_recommendations(self):
        """Generate actionable recommendations"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        model_age = self.get_model_age()
        
        recommendations = []
        
        if model_age > 30:
            recommendations.append("Model is over 30 days old - consider retraining")
        
        if latest_metrics and latest_metrics['roc_auc'] < 0.8:
            recommendations.append("Performance below 0.8 ROC-AUC - investigate data quality")
        
        if not recommendations:
            recommendations.append("No immediate action required")
        
        return recommendations

# Global monitor instance
model_monitor = None

def initialize_monitor():
    """Initialize the model monitor"""
    global model_monitor
    try:
        from utils import get_model_path
        model_path = get_model_path("pipeline_heart_optimized.joblib")
        data_path = get_model_path("../data/heart_clean.csv")
        model_monitor = ModelMonitor(model_path, data_path)
        logger.info("✅ Model monitoring system initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize model monitor: {e}")

if __name__ == "__main__":
    # Test the monitoring system
    initialize_monitor()
    if model_monitor:
        print("Model age:", model_monitor.get_model_age(), "days")