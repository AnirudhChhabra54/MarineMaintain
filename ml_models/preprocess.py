import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'engine_temperature',
            'fuel_level',
            'speed',
            'pressure',
            'vibration'
        ]
        self.categorical_columns = ['log_type', 'severity']

    def extract_numerical_features(self, content: str) -> Dict[str, float]:
        """Extract numerical features from log content."""
        try:
            features = {}
            # Try to parse content as JSON first
            try:
                data = json.loads(content)
                for feature in self.feature_columns:
                    if feature in data:
                        features[feature] = float(data[feature])
            except json.JSONDecodeError:
                # If not JSON, try regex pattern matching
                import re
                patterns = {
                    'engine_temperature': r'temperature:\s*([\d.]+)',
                    'fuel_level': r'fuel:\s*([\d.]+)',
                    'speed': r'speed:\s*([\d.]+)',
                    'pressure': r'pressure:\s*([\d.]+)',
                    'vibration': r'vibration:\s*([\d.]+)'
                }
                
                for feature, pattern in patterns.items():
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        features[feature] = float(match.group(1))

            return features

        except Exception as e:
            logger.error(f"Error extracting numerical features: {str(e)}")
            return {}

    def extract_temporal_features(self, timestamp: str) -> Dict[str, float]:
        """Extract temporal features from timestamp."""
        try:
            dt = pd.to_datetime(timestamp)
            return {
                'hour': dt.hour,
                'day_of_week': dt.dayofweek,
                'month': dt.month,
                'is_weekend': 1.0 if dt.dayofweek >= 5 else 0.0
            }
        except Exception as e:
            logger.error(f"Error extracting temporal features: {str(e)}")
            return {
                'hour': 0,
                'day_of_week': 0,
                'month': 0,
                'is_weekend': 0
            }

    def preprocess_single_log(self, log_entry: Dict[str, Any]) -> np.ndarray:
        """Preprocess a single log entry."""
        try:
            features = []
            
            # Extract numerical features from content
            numerical_features = self.extract_numerical_features(log_entry['content'])
            for feature in self.feature_columns:
                features.append(numerical_features.get(feature, 0.0))
            
            # Extract temporal features
            temporal_features = self.extract_temporal_features(log_entry['timestamp'])
            features.extend([
                temporal_features['hour'],
                temporal_features['day_of_week'],
                temporal_features['month'],
                temporal_features['is_weekend']
            ])
            
            # Add categorical features
            for col in self.categorical_columns:
                if col in log_entry:
                    encoded_value = self.label_encoder.transform([log_entry[col]])[0]
                    features.append(encoded_value)
                else:
                    features.append(0)
            
            return np.array(features)

        except Exception as e:
            logger.error(f"Error preprocessing single log: {str(e)}")
            return np.zeros(len(self.feature_columns) + 4 + len(self.categorical_columns))

    def fit_transform(self, logs: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """Fit preprocessor and transform log data."""
        try:
            # Initialize feature matrix
            n_features = len(self.feature_columns) + 4 + len(self.categorical_columns)
            X = np.zeros((len(logs), n_features))
            
            # Collect all categorical values for encoding
            for col in self.categorical_columns:
                unique_values = list(set(log[col] for log in logs if col in log))
                self.label_encoder.fit(unique_values)
            
            # Process each log entry
            for i, log in enumerate(logs):
                X[i] = self.preprocess_single_log(log)
            
            # Scale numerical features
            numerical_indices = list(range(len(self.feature_columns)))
            X[:, numerical_indices] = self.scaler.fit_transform(X[:, numerical_indices])
            
            # Create feature names for reference
            feature_names = (
                self.feature_columns +
                ['hour', 'day_of_week', 'month', 'is_weekend'] +
                self.categorical_columns
            )
            
            return X, feature_names

        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            return np.array([]), []

    def transform(self, logs: List[Dict[str, Any]]) -> np.ndarray:
        """Transform new log data using fitted preprocessor."""
        try:
            # Initialize feature matrix
            n_features = len(self.feature_columns) + 4 + len(self.categorical_columns)
            X = np.zeros((len(logs), n_features))
            
            # Process each log entry
            for i, log in enumerate(logs):
                X[i] = self.preprocess_single_log(log)
            
            # Scale numerical features
            numerical_indices = list(range(len(self.feature_columns)))
            X[:, numerical_indices] = self.scaler.transform(X[:, numerical_indices])
            
            return X

        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            return np.array([])

    def save_preprocessor(self, path: str):
        """Save preprocessor state."""
        try:
            import joblib
            state = {
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'categorical_columns': self.categorical_columns
            }
            joblib.dump(state, path)
            logger.info(f"Preprocessor saved to {path}")
        except Exception as e:
            logger.error(f"Error saving preprocessor: {str(e)}")

    @classmethod
    def load_preprocessor(cls, path: str) -> 'LogPreprocessor':
        """Load preprocessor state."""
        try:
            import joblib
            state = joblib.load(path)
            preprocessor = cls()
            preprocessor.scaler = state['scaler']
            preprocessor.label_encoder = state['label_encoder']
            preprocessor.feature_columns = state['feature_columns']
            preprocessor.categorical_columns = state['categorical_columns']
            logger.info(f"Preprocessor loaded from {path}")
            return preprocessor
        except Exception as e:
            logger.error(f"Error loading preprocessor: {str(e)}")
            return cls()

if __name__ == "__main__":
    # Example usage
    preprocessor = LogPreprocessor()
    
    # Sample log entries
    sample_logs = [
        {
            'timestamp': '2024-01-20 10:30:00',
            'content': '{"engine_temperature": 85.5, "fuel_level": 75.0, "speed": 12.5}',
            'log_type': 'status',
            'severity': 'normal'
        },
        {
            'timestamp': '2024-01-20 11:45:00',
            'content': '{"engine_temperature": 92.0, "fuel_level": 65.0, "speed": 15.0}',
            'log_type': 'warning',
            'severity': 'high'
        }
    ]
    
    # Fit and transform
    X, feature_names = preprocessor.fit_transform(sample_logs)
    print("Processed features shape:", X.shape)
    print("Feature names:", feature_names)
    
    # Save and load test
    preprocessor.save_preprocessor('preprocessor_state.joblib')
    loaded_preprocessor = LogPreprocessor.load_preprocessor('preprocessor_state.joblib')
    X_new = loaded_preprocessor.transform(sample_logs)
    print("Transformed features shape:", X_new.shape)
