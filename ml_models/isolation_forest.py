from sklearn.ensemble import IsolationForest
import numpy as np
import logging
from typing import Tuple, Optional
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShipLogIsolationForest:
    def __init__(self, 
                 n_estimators: int = 100,
                 contamination: float = 0.1,
                 random_state: int = 42):
        """
        Initialize Isolation Forest model for anomaly detection.
        
        Args:
            n_estimators: Number of base estimators in the ensemble
            contamination: Expected proportion of outliers in the dataset
            random_state: Random state for reproducibility
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.feature_names = None
        self.threshold = None
        self.scores_mean = None
        self.scores_std = None

    def fit(self, X: np.ndarray, feature_names: Optional[list] = None) -> 'ShipLogIsolationForest':
        """
        Fit the Isolation Forest model.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            feature_names: Optional list of feature names
        
        Returns:
            self: Returns the instance itself
        """
        try:
            logger.info(f"Training Isolation Forest with {X.shape[0]} samples")
            
            # Store feature names if provided
            self.feature_names = feature_names
            
            # Fit the model
            self.model.fit(X)
            
            # Compute decision scores for threshold calculation
            scores = self.model.score_samples(X)
            self.scores_mean = np.mean(scores)
            self.scores_std = np.std(scores)
            
            # Set threshold as mean - 2 standard deviations
            self.threshold = self.scores_mean - 2 * self.scores_std
            
            logger.info(f"Training completed. Score threshold set to: {self.threshold:.6f}")
            return self

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomaly scores and labels for input data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        
        Returns:
            Tuple containing:
                - Anomaly scores (lower = more anomalous)
                - Binary labels (1 = normal, -1 = anomaly)
        """
        try:
            logger.info(f"Predicting anomalies for {X.shape[0]} samples")
            
            # Get anomaly scores
            scores = self.model.score_samples(X)
            
            # Get binary predictions based on threshold
            predictions = np.where(scores > self.threshold, 1, -1)
            
            # Calculate proportion of anomalies
            anomaly_ratio = np.mean(predictions == -1)
            logger.info(f"Detected {anomaly_ratio:.2%} anomalies")
            
            return scores, predictions

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return np.array([]), np.array([])

    def get_feature_importance(self) -> dict:
        """
        Calculate feature importance scores based on average path length.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            if not self.feature_names:
                logger.warning("Feature names not provided during training")
                return {}

            # Calculate average path length for each feature
            importances = np.zeros(len(self.feature_names))
            
            for tree in self.model.estimators_:
                # Get path lengths for each feature
                paths = tree.decision_path(np.eye(len(self.feature_names)))
                importances += np.array([len(path.indices) for path in paths])
            
            importances /= self.model.n_estimators_
            
            # Normalize importances
            importances = (importances - np.min(importances)) / (np.max(importances) - np.min(importances))
            
            # Create dictionary of feature importances
            importance_dict = dict(zip(self.feature_names, importances))
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict

        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return {}

    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        try:
            state = {
                'model': self.model,
                'feature_names': self.feature_names,
                'threshold': self.threshold,
                'scores_mean': self.scores_mean,
                'scores_std': self.scores_std
            }
            joblib.dump(state, path)
            logger.info(f"Model saved to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    @classmethod
    def load_model(cls, path: str) -> Optional['ShipLogIsolationForest']:
        """
        Load a saved model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded ShipLogIsolationForest instance
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")

            state = joblib.load(path)
            
            instance = cls()
            instance.model = state['model']
            instance.feature_names = state['feature_names']
            instance.threshold = state['threshold']
            instance.scores_mean = state['scores_mean']
            instance.scores_std = state['scores_std']
            
            logger.info(f"Model loaded from {path}")
            return instance

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None

    def get_model_info(self) -> dict:
        """
        Get information about the trained model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'n_estimators': self.model.n_estimators,
            'contamination': self.model.contamination,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'threshold': self.threshold,
            'scores_mean': self.scores_mean,
            'scores_std': self.scores_std
        }

if __name__ == "__main__":
    # Example usage
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    
    # Add some obvious anomalies
    X[-10:] = X[-10:] * 5
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Initialize and train model
    model = ShipLogIsolationForest(n_estimators=100, contamination=0.1)
    model.fit(X, feature_names)
    
    # Make predictions
    scores, predictions = model.predict(X)
    print(f"Detected {np.sum(predictions == -1)} anomalies")
    
    # Get feature importance
    importance = model.get_feature_importance()
    print("\nFeature Importance:")
    for feature, score in importance.items():
        print(f"{feature}: {score:.4f}")
    
    # Save and load test
    model.save_model('isolation_forest_model.joblib')
    loaded_model = ShipLogIsolationForest.load_model('isolation_forest_model.joblib')
    if loaded_model:
        new_scores, new_predictions = loaded_model.predict(X)
        print(f"\nLoaded model detected {np.sum(new_predictions == -1)} anomalies")
