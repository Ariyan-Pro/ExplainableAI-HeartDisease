# healthcare_model/data_validation.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pydantic import BaseModel, validator
import json

logger = logging.getLogger(__name__)

class DataValidator:
    """Advanced data validation pipeline for medical data"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self):
        """Load medical data validation rules"""
        rules = {
            'age': {'min': 1, 'max': 120, 'type': 'int'},
            'sex': {'allowed_values': [0, 1], 'type': 'int'},
            'cp': {'min': 0, 'max': 3, 'type': 'int'},
            'trestbps': {'min': 50, 'max': 250, 'type': 'int'},
            'chol': {'min': 100, 'max': 600, 'type': 'int'},
            'fbs': {'allowed_values': [0, 1], 'type': 'int'},
            'restecg': {'min': 0, 'max': 2, 'type': 'int'},
            'thalach': {'min': 50, 'max': 220, 'type': 'int'},
            'exang': {'allowed_values': [0, 1], 'type': 'int'},
            'oldpeak': {'min': 0.0, 'max': 10.0, 'type': 'float'},
            'slope': {'min': 0, 'max': 2, 'type': 'int'},
            'ca': {'min': 0, 'max': 3, 'type': 'int'},
            'thal': {'min': 1, 'max': 3, 'type': 'int'}
        }
        return rules
    
    def validate_single_record(self, record: dict) -> Tuple[bool, List[str]]:
        """Validate a single patient record"""
        errors = []
        
        for field, value in record.items():
            if field not in self.validation_rules:
                errors.append(f"Unknown field: {field}")
                continue
            
            rules = self.validation_rules[field]
            
            # Type validation
            try:
                if rules['type'] == 'int':
                    value = int(value)
                elif rules['type'] == 'float':
                    value = float(value)
            except (ValueError, TypeError):
                errors.append(f"Invalid type for {field}: expected {rules['type']}")
                continue
            
            # Range validation
            if 'min' in rules and 'max' in rules:
                if not (rules['min'] <= value <= rules['max']):
                    errors.append(f"{field} out of range: {value} not in [{rules['min']}, {rules['max']}]")
            
            # Allowed values validation
            if 'allowed_values' in rules:
                if value not in rules['allowed_values']:
                    errors.append(f"{field} has invalid value: {value}, allowed: {rules['allowed_values']}")
        
        return len(errors) == 0, errors
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict:
        """Validate entire dataset with comprehensive checks"""
        validation_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_records': len(df),
            'valid_records': 0,
            'invalid_records': 0,
            'field_validation': {},
            'data_quality_metrics': {},
            'errors': []
        }
        
        # Field-level validation
        for column in df.columns:
            if column in self.validation_rules:
                rules = self.validation_rules[column]
                validation_report['field_validation'][column] = {
                    'missing_values': df[column].isna().sum(),
                    'out_of_range': self._count_out_of_range(df[column], rules),
                    'invalid_types': self._count_invalid_types(df[column], rules)
                }
        
        # Record-level validation
        valid_records = 0
        for idx, record in df.iterrows():
            is_valid, errors = self.validate_single_record(record.to_dict())
            if is_valid:
                valid_records += 1
            else:
                validation_report['errors'].append({
                    'record_index': idx,
                    'errors': errors
                })
        
        validation_report['valid_records'] = valid_records
        validation_report['invalid_records'] = len(df) - valid_records
        
        # Data quality metrics
        validation_report['data_quality_metrics'] = {
            'completeness_rate': valid_records / len(df) if len(df) > 0 else 0,
            'field_completeness': {col: 1 - (df[col].isna().sum() / len(df)) for col in df.columns},
            'expected_ranges_conformance': self._calculate_range_conformance(df)
        }
        
        logger.info(f"Data validation completed: {valid_records}/{len(df)} valid records")
        return validation_report
    
    def _count_out_of_range(self, series: pd.Series, rules: dict) -> int:
        """Count values outside allowed range"""
        if 'min' not in rules or 'max' not in rules:
            return 0
        
        try:
            if rules['type'] == 'int':
                series = pd.to_numeric(series, errors='coerce')
            return ((series < rules['min']) | (series > rules['max'])).sum()
        except:
            return len(series)
    
    def _count_invalid_types(self, series: pd.Series, rules: dict) -> int:
        """Count values with invalid types"""
        try:
            if rules['type'] == 'int':
                pd.to_numeric(series, errors='coerce').astype(int)
                return series.isna().sum()  # NaN indicates conversion failure
            elif rules['type'] == 'float':
                pd.to_numeric(series, errors='coerce')
                return series.isna().sum()
        except:
            return len(series)
        return 0
    
    def _calculate_range_conformance(self, df: pd.DataFrame) -> Dict:
        """Calculate how well data conforms to expected ranges"""
        conformance = {}
        
        for column in df.columns:
            if column in self.validation_rules:
                rules = self.validation_rules[column]
                if 'min' in rules and 'max' in rules:
                    valid_count = ((df[column] >= rules['min']) & (df[column] <= rules['max'])).sum()
                    conformance[column] = valid_count / len(df) if len(df) > 0 else 0
        
        return conformance
    
    def generate_validation_report(self, df: pd.DataFrame) -> str:
        """Generate human-readable validation report"""
        validation_result = self.validate_dataset(df)
        
        report_lines = [
            "DATA VALIDATION REPORT",
            "=" * 50,
            f"Timestamp: {validation_result['timestamp']}",
            f"Total Records: {validation_result['total_records']}",
            f"Valid Records: {validation_result['valid_records']}",
            f"Invalid Records: {validation_result['invalid_records']}",
            f"Data Quality Score: {validation_result['data_quality_metrics']['completeness_rate']:.1%}",
            "",
            "FIELD-LEVEL VALIDATION:"
        ]
        
        for field, stats in validation_result['field_validation'].items():
            report_lines.append(
                f"  {field}: {stats['missing_values']} missing, "
                f"{stats['out_of_range']} out-of-range, "
                f"{stats['invalid_types']} type errors"
            )
        
        if validation_result['errors']:
            report_lines.extend(["", "DETAILED ERRORS:"])
            for error in validation_result['errors'][:5]:  # Show first 5 errors
                report_lines.append(f"  Record {error['record_index']}: {', '.join(error['errors'][:2])}")
            if len(validation_result['errors']) > 5:
                report_lines.append(f"  ... and {len(validation_result['errors']) - 5} more errors")
        
        return "\n".join(report_lines)

# Global validator instance
data_validator = DataValidator()

def validate_incoming_data(data: dict) -> Tuple[bool, List[str]]:
    """Validate incoming API data"""
    return data_validator.validate_single_record(data)

def validate_training_data(df: pd.DataFrame) -> Dict:
    """Validate training dataset"""
    return data_validator.validate_dataset(df)

if __name__ == "__main__":
    # Test the data validation
    from utils import load_data
    
    df = load_data().drop(columns=['target'])
    report = data_validator.generate_validation_report(df)
    print(report)