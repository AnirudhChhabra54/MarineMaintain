import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
import os
from datetime import datetime
from preprocess import LogPreprocessor
from autoencoder import ShipLogAutoencoder
from isolation_forest import ShipLogIsolationForest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, model_dir: str, timestamp: str):
        """
        Initialize the anomaly detector with trained models.
        
        Args:
            model_dir: Directory containing saved models
            timestamp: Timestamp of the models to load
        """
        self.model_dir = model_dir
        self.timestamp = timestamp
        
        # Model paths
        self.model_paths = {
            'preprocessor': os.path.join(model_dir, f'preprocessor_{timestamp}.joblib'),
            'autoencoder': os.path.join(model_dir, f'autoencoder_{timestamp}.pth'),
            'isolation_forest': os.path.join(model_dir, f'isolation_forest_{timestamp}.joblib')
        }
        
        # Load models
        self.preprocessor = None
        self.autoencoder = None
        self.isolation_forest = None
        self.load_models()
        
        # Ensemble weights
        self.weights = {
            'autoencoder': 0.6,
            'isolation_forest': 0.4
        }

    def load_models(self):
        """Load all required models."""
        try:
            # Load preprocessor
            self.preprocessor = LogPreprocessor.load_preprocessor(
                self.model_paths['preprocessor']
            )
            if not self.preprocessor:
                raise ValueError("Failed to load preprocessor")

            # Load autoencoder
            self.autoencoder = ShipLogAutoencoder.load_model(
                self.model_paths['autoencoder']
            )
            if not self.autoencoder:
                raise ValueError("Failed to load autoencoder")

            # Load isolation forest
            self.isolation_forest = ShipLogIsolationForest.load_model(
                self.model_paths['isolation_forest']
            )
            if not self.isolation_forest:
                raise ValueError("Failed to load isolation forest")

            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def preprocess_logs(self, logs: List[Dict[str, Any]]) -> np.ndarray:
        """
        Preprocess log entries for inference.
        
        Args:
            logs: List of log entries
            
        Returns:
            Preprocessed features as numpy array
        """
        try:
            return self.preprocessor.transform(logs)
        except Exception as e:
            logger.error(f"Error preprocessing logs: {str(e)}")
            raise

    def detect_anomalies(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in the provided logs using ensemble approach.
        
        Args:
            logs: List of log entries
            
        Returns:
            List of logs with anomaly scores and predictions
        """
        try:
            # Preprocess logs
            X = self.preprocess_logs(logs)
            if len(X) == 0:
                return []

            # Get predictions from both models
            ae_scores, ae_anomalies = self.autoencoder.predict(X)
            if_scores, if_predictions = self.isolation_forest.predict(X)

            # Normalize scores to [0, 1] range
            ae_scores_norm = (ae_scores - np.min(ae_scores)) / (np.max(ae_scores) - np.min(ae_scores))
            if_scores_norm = (if_scores - np.min(if_scores)) / (np.max(if_scores) - np.min(if_scores))

            # Combine scores using weighted average
            ensemble_scores = (
                self.weights['autoencoder'] * ae_scores_norm +
                self.weights['isolation_forest'] * if_scores_norm
            )

            # Determine final predictions
            # Use mean + 2*std as threshold for ensemble scores
            threshold = np.mean(ensemble_scores) + 2 * np.std(ensemble_scores)
            ensemble_predictions = ensemble_scores > threshold

            # Prepare results
            results = []
            for i, log in enumerate(logs):
                result = log.copy()
                result.update({
                    'anomaly_score': float(ensemble_scores[i]),
                    'is_anomaly': bool(ensemble_predictions[i]),
                    'autoencoder_score': float(ae_scores[i]),
                    'isolation_forest_score': float(if_scores[i]),
                    'detection_timestamp': datetime.now().isoformat(),
                    'severity': 'high' if ensemble_predictions[i] else 'normal'
                })
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return []

    def analyze_anomaly(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform detailed analysis of a single anomalous log entry.
        
        Args:
            log: Single log entry
            
        Returns:
            Dictionary containing detailed analysis
        """
        try:
            # Preprocess single log
            X = self.preprocess_logs([log])
            if len(X) == 0:
                return {}

            # Get feature importance from isolation forest
            feature_importance = self.isolation_forest.get_feature_importance()

            # Get reconstruction error details from autoencoder
            reconstructed = self.autoencoder.model(
                torch.FloatTensor(X).to(self.autoencoder.device)
            ).detach().cpu().numpy()
            
            feature_errors = np.abs(X - reconstructed)[0]

            # Combine analyses
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'log_id': log.get('id', 'unknown'),
                'feature_importance': feature_importance,
                'reconstruction_errors': {
                    feature: float(error)
                    for feature, error in zip(self.preprocessor.feature_columns, feature_errors)
                },
                'contributing_factors': [],
                'recommendations': []
            }

            # Identify main contributing factors
            sorted_features = sorted(
                zip(self.preprocessor.feature_columns, feature_errors),
                key=lambda x: x[1],
                reverse=True
            )

            # Add top contributing factors
            for feature, error in sorted_features[:3]:
                analysis['contributing_factors'].append({
                    'feature': feature,
                    'error': float(error),
                    'importance': float(feature_importance.get(feature, 0))
                })

            # Generate recommendations
            for factor in analysis['contributing_factors']:
                recommendation = self._generate_recommendation(
                    factor['feature'],
                    factor['error'],
                    log
                )
                if recommendation:
                    analysis['recommendations'].append(recommendation)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing anomaly: {str(e)}")
            return {}

    def _generate_recommendation(self, feature: str, error: float, log: Dict[str, Any]) -> str:
        """Generate recommendation based on the anomalous feature."""
        try:
            base_recommendations = {
                'engine_temperature': "Check engine cooling system and perform temperature regulation",
                'fuel_level': "Inspect fuel system and verify fuel consumption rate",
                'speed': "Review speed logs and check for navigation system issues",
                'pressure': "Examine pressure control systems and verify sensor calibration",
                'vibration': "Inspect for mechanical issues and perform vibration analysis"
            }

            recommendation = base_recommendations.get(feature, "")
            if recommendation:
                severity = "high" if error > 0.8 else "medium" if error > 0.5 else "low"
                return f"{recommendation} (Severity: {severity})"
            return ""

        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return ""

    def get_model_health(self) -> Dict[str, Any]:
        """
        Check the health status of the models.
        
        Returns:
            Dictionary containing model health metrics
        """
        try:
            return {
                'status': 'healthy',
                'last_checked': datetime.now().isoformat(),
                'models': {
                    'preprocessor': {
                        'loaded': self.preprocessor is not None,
                        'features': len(self.preprocessor.feature_columns) if self.preprocessor else 0
                    },
                    'autoencoder': {
                        'loaded': self.autoencoder is not None,
                        'threshold': float(self.autoencoder.threshold) if self.autoencoder else None
                    },
                    'isolation_forest': {
                        'loaded': self.isolation_forest is not None,
                        'threshold': float(self.isolation_forest.threshold) if self.isolation_forest else None
                    }
                },
                'weights': self.weights
            }
        except Exception as e:
            logger.error(f"Error checking model health: {str(e)}")
            return {'status': 'unhealthy', 'error': str(e)}

if __name__ == "__main__":
    # Example usage
    
    # Initialize detector with saved models
    detector = AnomalyDetector('models', '20240120_120000')
    
    # Sample logs
    sample_logs = [
        {
            'timestamp': datetime.now().isoformat(),
            'content': '{"engine_temperature": 95.5, "fuel_level": 45.0, "speed": 18.5}',
            'log_type': 'status',
            'severity': 'normal'
        },
        {
            'timestamp': datetime.now().isoformat(),
            'content': '{"engine_temperature": 150.0, "fuel_level": 10.0, "speed": 25.0}',
            'log_type': 'warning',
            'severity': 'high'
        }
    ]
    
    try:
        # Detect anomalies
        results = detector.detect_anomalies(sample_logs)
        
        print("\nAnomaly Detection Results:")
        for result in results:
            print(f"\nLog Type: {result['log_type']}")
            print(f"Anomaly Score: {result['anomaly_score']:.4f}")
            print(f"Is Anomaly: {result['is_anomaly']}")
            print(f"Severity: {result['severity']}")
        
        # Analyze anomalous log
        for result in results:
            if result['is_anomaly']:
                analysis = detector.analyze_anomaly(result)
                print(f"\nDetailed Analysis for Anomalous Log:")
                print(f"Contributing Factors:")
                for factor in analysis['contributing_factors']:
                    print(f"- {factor['feature']}: Error = {factor['error']:.4f}")
                print("\nRecommendations:")
                for rec in analysis['recommendations']:
                    print(f"- {rec}")
        
        # Check model health
        health = detector.get_model_health()
        print("\nModel Health Status:", health['status'])
    
    except Exception as e:
        print(f"Error in example: {str(e)}")
