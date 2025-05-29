import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional
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

class ModelTrainer:
    def __init__(self, 
                 model_dir: str = 'models',
                 autoencoder_encoding_dim: int = 10,
                 isolation_forest_estimators: int = 100):
        """
        Initialize the model trainer.
        
        Args:
            model_dir: Directory to save trained models
            autoencoder_encoding_dim: Dimension of autoencoder's encoded representation
            isolation_forest_estimators: Number of estimators for Isolation Forest
        """
        self.model_dir = model_dir
        self.autoencoder_encoding_dim = autoencoder_encoding_dim
        self.isolation_forest_estimators = isolation_forest_estimators
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.preprocessor = LogPreprocessor()
        self.autoencoder = None
        self.isolation_forest = None
        
        # Paths for saving models
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_paths = {
            'preprocessor': os.path.join(model_dir, f'preprocessor_{self.timestamp}.joblib'),
            'autoencoder': os.path.join(model_dir, f'autoencoder_{self.timestamp}.pth'),
            'isolation_forest': os.path.join(model_dir, f'isolation_forest_{self.timestamp}.joblib')
        }

    def prepare_data(self, logs: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare data for training.
        
        Args:
            logs: List of log entries
            
        Returns:
            Tuple of processed features and feature names
        """
        try:
            logger.info("Preparing data for training...")
            X, feature_names = self.preprocessor.fit_transform(logs)
            logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
            return X, feature_names

        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def train_models(self, X: np.ndarray, feature_names: List[str]) -> Tuple[dict, dict]:
        """
        Train both autoencoder and isolation forest models.
        
        Args:
            X: Preprocessed features
            feature_names: List of feature names
            
        Returns:
            Tuple of training metrics for both models
        """
        try:
            logger.info("Starting model training...")
            
            # Train Autoencoder
            logger.info("Training Autoencoder...")
            self.autoencoder = ShipLogAutoencoder(
                input_dim=X.shape[1],
                encoding_dim=self.autoencoder_encoding_dim
            )
            train_losses, val_losses = self.autoencoder.fit(X)
            
            # Train Isolation Forest
            logger.info("Training Isolation Forest...")
            self.isolation_forest = ShipLogIsolationForest(
                n_estimators=self.isolation_forest_estimators
            )
            self.isolation_forest.fit(X, feature_names)
            
            # Get feature importance from Isolation Forest
            feature_importance = self.isolation_forest.get_feature_importance()
            
            # Compile metrics
            autoencoder_metrics = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'final_train_loss': train_losses[-1],
                'final_val_loss': val_losses[-1]
            }
            
            isolation_forest_metrics = {
                'feature_importance': feature_importance,
                'model_info': self.isolation_forest.get_model_info()
            }
            
            return autoencoder_metrics, isolation_forest_metrics

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise

    def save_models(self):
        """Save all trained models and preprocessor."""
        try:
            logger.info("Saving models...")
            
            # Save preprocessor
            self.preprocessor.save_preprocessor(self.model_paths['preprocessor'])
            
            # Save autoencoder
            if self.autoencoder:
                self.autoencoder.save_model(self.model_paths['autoencoder'])
            
            # Save isolation forest
            if self.isolation_forest:
                self.isolation_forest.save_model(self.model_paths['isolation_forest'])
            
            logger.info("All models saved successfully")

        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
            raise

    @classmethod
    def load_models(cls, model_dir: str, timestamp: str) -> Optional['ModelTrainer']:
        """
        Load saved models.
        
        Args:
            model_dir: Directory containing saved models
            timestamp: Timestamp of the models to load
            
        Returns:
            ModelTrainer instance with loaded models
        """
        try:
            trainer = cls(model_dir=model_dir)
            trainer.timestamp = timestamp
            
            # Update model paths
            trainer.model_paths = {
                'preprocessor': os.path.join(model_dir, f'preprocessor_{timestamp}.joblib'),
                'autoencoder': os.path.join(model_dir, f'autoencoder_{timestamp}.pth'),
                'isolation_forest': os.path.join(model_dir, f'isolation_forest_{timestamp}.joblib')
            }
            
            # Load models
            trainer.preprocessor = LogPreprocessor.load_preprocessor(
                trainer.model_paths['preprocessor']
            )
            trainer.autoencoder = ShipLogAutoencoder.load_model(
                trainer.model_paths['autoencoder']
            )
            trainer.isolation_forest = ShipLogIsolationForest.load_model(
                trainer.model_paths['isolation_forest']
            )
            
            if all([trainer.preprocessor, trainer.autoencoder, trainer.isolation_forest]):
                logger.info("All models loaded successfully")
                return trainer
            else:
                logger.error("Failed to load one or more models")
                return None

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return None

    def evaluate_models(self, X: np.ndarray) -> dict:
        """
        Evaluate both models on test data.
        
        Args:
            X: Test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            metrics = {}
            
            # Autoencoder evaluation
            if self.autoencoder:
                ae_scores, ae_anomalies = self.autoencoder.predict(X)
                metrics['autoencoder'] = {
                    'anomaly_ratio': float(np.mean(ae_anomalies)),
                    'mean_reconstruction_error': float(np.mean(ae_scores)),
                    'std_reconstruction_error': float(np.std(ae_scores))
                }
            
            # Isolation Forest evaluation
            if self.isolation_forest:
                if_scores, if_predictions = self.isolation_forest.predict(X)
                metrics['isolation_forest'] = {
                    'anomaly_ratio': float(np.mean(if_predictions == -1)),
                    'mean_score': float(np.mean(if_scores)),
                    'std_score': float(np.std(if_scores))
                }
            
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating models: {str(e)}")
            return {}

if __name__ == "__main__":
    # Example usage
    
    # Generate sample data
    n_samples = 1000
    n_features = 20
    X = np.random.randn(n_samples, n_features)
    
    # Add some anomalies
    X[-50:] = X[-50:] * 3
    
    # Create sample logs
    sample_logs = []
    for i in range(n_samples):
        log = {
            'timestamp': datetime.now().isoformat(),
            'content': f"Sample log {i}",
            'log_type': 'status',
            'severity': 'normal'
        }
        sample_logs.append(log)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    try:
        # Prepare data
        X_processed, feature_names = trainer.prepare_data(sample_logs)
        
        # Train models
        ae_metrics, if_metrics = trainer.train_models(X_processed, feature_names)
        
        print("\nAutoencoder Metrics:")
        print(f"Final training loss: {ae_metrics['final_train_loss']:.6f}")
        print(f"Final validation loss: {ae_metrics['final_val_loss']:.6f}")
        
        print("\nIsolation Forest Feature Importance:")
        for feature, importance in if_metrics['feature_importance'].items():
            print(f"{feature}: {importance:.4f}")
        
        # Save models
        trainer.save_models()
        
        # Load models
        loaded_trainer = ModelTrainer.load_models(trainer.model_dir, trainer.timestamp)
        
        if loaded_trainer:
            # Evaluate models
            eval_metrics = loaded_trainer.evaluate_models(X_processed)
            
            print("\nEvaluation Metrics:")
            print(f"Autoencoder anomaly ratio: {eval_metrics['autoencoder']['anomaly_ratio']:.2%}")
            print(f"Isolation Forest anomaly ratio: {eval_metrics['isolation_forest']['anomaly_ratio']:.2%}")
    
    except Exception as e:
        print(f"Error in example: {str(e)}")
