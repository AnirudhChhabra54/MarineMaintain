import pytest
import numpy as np
import torch
import os
from datetime import datetime
from ml_models.preprocess import LogPreprocessor
from ml_models.autoencoder import ShipLogAutoencoder
from ml_models.isolation_forest import ShipLogIsolationForest
from ml_models.train import ModelTrainer
from ml_models.inference import AnomalyDetector

@pytest.fixture
def sample_logs():
    """Generate sample log data for testing"""
    logs = []
    for i in range(100):
        # Generate normal logs
        if i < 80:
            temperature = np.random.normal(85, 5)  # Normal temperature range
            fuel = np.random.normal(75, 10)       # Normal fuel range
            speed = np.random.normal(12, 2)       # Normal speed range
        # Generate anomalous logs
        else:
            temperature = np.random.normal(150, 10)  # Abnormal temperature
            fuel = np.random.normal(20, 5)          # Low fuel
            speed = np.random.normal(25, 3)         # High speed

        log = {
            'timestamp': datetime.now().isoformat(),
            'ship_id': f'VSL{i:03d}',
            'log_type': 'status',
            'content': {
                'engine_temperature': float(temperature),
                'fuel_level': float(fuel),
                'speed': float(speed),
                'pressure': float(np.random.normal(2, 0.5)),
                'vibration': float(np.random.normal(0.3, 0.1))
            },
            'severity': 'normal' if i < 80 else 'high'
        }
        logs.append(log)
    return logs

@pytest.fixture
def model_paths(tmp_path):
    """Create temporary directory for model artifacts"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        'dir': model_dir,
        'timestamp': timestamp,
        'preprocessor': model_dir / f"preprocessor_{timestamp}.joblib",
        'autoencoder': model_dir / f"autoencoder_{timestamp}.pth",
        'isolation_forest': model_dir / f"isolation_forest_{timestamp}.joblib"
    }

class TestPreprocessor:
    def test_initialization(self):
        preprocessor = LogPreprocessor()
        assert preprocessor.feature_columns is not None
        assert preprocessor.categorical_columns is not None

    def test_feature_extraction(self, sample_logs):
        preprocessor = LogPreprocessor()
        
        # Test numerical feature extraction
        content = sample_logs[0]['content']
        features = preprocessor.extract_numerical_features(str(content))
        assert 'engine_temperature' in features
        assert 'fuel_level' in features
        assert 'speed' in features

    def test_temporal_features(self, sample_logs):
        preprocessor = LogPreprocessor()
        
        # Test temporal feature extraction
        temporal_features = preprocessor.extract_temporal_features(sample_logs[0]['timestamp'])
        assert all(key in temporal_features for key in ['hour', 'day_of_week', 'month', 'is_weekend'])

    def test_fit_transform(self, sample_logs):
        preprocessor = LogPreprocessor()
        
        # Test full preprocessing pipeline
        X, feature_names = preprocessor.fit_transform(sample_logs)
        assert X.shape[0] == len(sample_logs)
        assert len(feature_names) > 0
        assert X.shape[1] == len(feature_names)

    def test_save_load(self, sample_logs, model_paths):
        preprocessor = LogPreprocessor()
        X, _ = preprocessor.fit_transform(sample_logs)
        
        # Test save
        preprocessor.save_preprocessor(model_paths['preprocessor'])
        assert os.path.exists(model_paths['preprocessor'])
        
        # Test load
        loaded_preprocessor = LogPreprocessor.load_preprocessor(model_paths['preprocessor'])
        assert loaded_preprocessor is not None
        
        # Test transform with loaded preprocessor
        X_loaded = loaded_preprocessor.transform(sample_logs)
        assert np.allclose(X, X_loaded)

class TestAutoencoder:
    def test_initialization(self):
        autoencoder = ShipLogAutoencoder(input_dim=20)
        assert isinstance(autoencoder.model, torch.nn.Module)
        assert autoencoder.device in ['cuda', 'cpu']

    def test_training(self, sample_logs):
        # Prepare data
        preprocessor = LogPreprocessor()
        X, _ = preprocessor.fit_transform(sample_logs)
        
        # Train autoencoder
        autoencoder = ShipLogAutoencoder(input_dim=X.shape[1])
        train_losses, val_losses = autoencoder.fit(X)
        
        assert len(train_losses) > 0
        assert len(val_losses) > 0
        assert train_losses[-1] < train_losses[0]  # Loss should decrease

    def test_prediction(self, sample_logs):
        # Prepare data
        preprocessor = LogPreprocessor()
        X, _ = preprocessor.fit_transform(sample_logs)
        
        # Train and predict
        autoencoder = ShipLogAutoencoder(input_dim=X.shape[1])
        autoencoder.fit(X)
        scores, anomalies = autoencoder.predict(X)
        
        assert len(scores) == len(X)
        assert len(anomalies) == len(X)
        assert np.sum(anomalies) > 0  # Should detect some anomalies

    def test_save_load(self, sample_logs, model_paths):
        # Prepare data
        preprocessor = LogPreprocessor()
        X, _ = preprocessor.fit_transform(sample_logs)
        
        # Train and save
        autoencoder = ShipLogAutoencoder(input_dim=X.shape[1])
        autoencoder.fit(X)
        autoencoder.save_model(model_paths['autoencoder'])
        
        # Load and verify
        loaded_model = ShipLogAutoencoder.load_model(model_paths['autoencoder'])
        assert loaded_model is not None
        
        scores1, anomalies1 = autoencoder.predict(X)
        scores2, anomalies2 = loaded_model.predict(X)
        assert np.allclose(scores1, scores2)
        assert np.array_equal(anomalies1, anomalies2)

class TestIsolationForest:
    def test_initialization(self):
        model = ShipLogIsolationForest()
        assert model.model is not None

    def test_training(self, sample_logs):
        # Prepare data
        preprocessor = LogPreprocessor()
        X, feature_names = preprocessor.fit_transform(sample_logs)
        
        # Train model
        model = ShipLogIsolationForest()
        model.fit(X, feature_names)
        
        assert model.threshold is not None
        assert model.scores_mean is not None
        assert model.scores_std is not None

    def test_prediction(self, sample_logs):
        # Prepare data
        preprocessor = LogPreprocessor()
        X, feature_names = preprocessor.fit_transform(sample_logs)
        
        # Train and predict
        model = ShipLogIsolationForest()
        model.fit(X, feature_names)
        scores, predictions = model.predict(X)
        
        assert len(scores) == len(X)
        assert len(predictions) == len(X)
        assert np.sum(predictions == -1) > 0  # Should detect some anomalies

    def test_feature_importance(self, sample_logs):
        # Prepare data
        preprocessor = LogPreprocessor()
        X, feature_names = preprocessor.fit_transform(sample_logs)
        
        # Train and get feature importance
        model = ShipLogIsolationForest()
        model.fit(X, feature_names)
        importance = model.get_feature_importance()
        
        assert len(importance) == len(feature_names)
        assert all(0 <= score <= 1 for score in importance.values())

    def test_save_load(self, sample_logs, model_paths):
        # Prepare data
        preprocessor = LogPreprocessor()
        X, feature_names = preprocessor.fit_transform(sample_logs)
        
        # Train and save
        model = ShipLogIsolationForest()
        model.fit(X, feature_names)
        model.save_model(model_paths['isolation_forest'])
        
        # Load and verify
        loaded_model = ShipLogIsolationForest.load_model(model_paths['isolation_forest'])
        assert loaded_model is not None
        
        scores1, pred1 = model.predict(X)
        scores2, pred2 = loaded_model.predict(X)
        assert np.allclose(scores1, scores2)
        assert np.array_equal(pred1, pred2)

class TestModelTrainer:
    def test_initialization(self, model_paths):
        trainer = ModelTrainer(model_dir=str(model_paths['dir']))
        assert trainer.preprocessor is not None
        assert trainer.model_dir == str(model_paths['dir'])

    def test_data_preparation(self, sample_logs):
        trainer = ModelTrainer()
        X, feature_names = trainer.prepare_data(sample_logs)
        
        assert X.shape[0] == len(sample_logs)
        assert len(feature_names) > 0

    def test_model_training(self, sample_logs):
        trainer = ModelTrainer()
        X, feature_names = trainer.prepare_data(sample_logs)
        
        ae_metrics, if_metrics = trainer.train_models(X, feature_names)
        
        assert 'train_losses' in ae_metrics
        assert 'feature_importance' in if_metrics

    def test_save_load(self, sample_logs, model_paths):
        # Train and save
        trainer = ModelTrainer(model_dir=str(model_paths['dir']))
        X, feature_names = trainer.prepare_data(sample_logs)
        trainer.train_models(X, feature_names)
        trainer.save_models()
        
        # Load and verify
        loaded_trainer = ModelTrainer.load_models(
            str(model_paths['dir']),
            trainer.timestamp
        )
        assert loaded_trainer is not None
        
        # Test evaluation
        metrics = loaded_trainer.evaluate_models(X)
        assert 'autoencoder' in metrics
        assert 'isolation_forest' in metrics

class TestAnomalyDetector:
    def test_initialization(self, sample_logs, model_paths):
        # First train and save models
        trainer = ModelTrainer(model_dir=str(model_paths['dir']))
        X, feature_names = trainer.prepare_data(sample_logs)
        trainer.train_models(X, feature_names)
        trainer.save_models()
        
        # Initialize detector
        detector = AnomalyDetector(str(model_paths['dir']), trainer.timestamp)
        assert detector.preprocessor is not None
        assert detector.autoencoder is not None
        assert detector.isolation_forest is not None

    def test_anomaly_detection(self, sample_logs, model_paths):
        # Train and save models
        trainer = ModelTrainer(model_dir=str(model_paths['dir']))
        X, feature_names = trainer.prepare_data(sample_logs)
        trainer.train_models(X, feature_names)
        trainer.save_models()
        
        # Test detection
        detector = AnomalyDetector(str(model_paths['dir']), trainer.timestamp)
        results = detector.detect_anomalies(sample_logs)
        
        assert len(results) == len(sample_logs)
        assert all('anomaly_score' in result for result in results)
        assert all('is_anomaly' in result for result in results)

    def test_anomaly_analysis(self, sample_logs, model_paths):
        # Train and save models
        trainer = ModelTrainer(model_dir=str(model_paths['dir']))
        X, feature_names = trainer.prepare_data(sample_logs)
        trainer.train_models(X, feature_names)
        trainer.save_models()
        
        # Test analysis
        detector = AnomalyDetector(str(model_paths['dir']), trainer.timestamp)
        results = detector.detect_anomalies(sample_logs)
        
        # Analyze first anomaly
        anomaly = next(result for result in results if result['is_anomaly'])
        analysis = detector.analyze_anomaly(anomaly)
        
        assert 'contributing_factors' in analysis
        assert 'recommendations' in analysis
        assert len(analysis['contributing_factors']) > 0

    def test_model_health(self, sample_logs, model_paths):
        # Train and save models
        trainer = ModelTrainer(model_dir=str(model_paths['dir']))
        X, feature_names = trainer.prepare_data(sample_logs)
        trainer.train_models(X, feature_names)
        trainer.save_models()
        
        # Test health check
        detector = AnomalyDetector(str(model_paths['dir']), trainer.timestamp)
        health = detector.get_model_health()
        
        assert health['status'] == 'healthy'
        assert 'models' in health
        assert all(model in health['models'] for model in ['preprocessor', 'autoencoder', 'isolation_forest'])

if __name__ == "__main__":
    pytest.main([__file__])
