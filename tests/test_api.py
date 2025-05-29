import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json
from backend.main import app

client = TestClient(app)

@pytest.fixture
def sample_log():
    """Fixture for a sample log entry"""
    return {
        "ship_id": "VSL001",
        "timestamp": datetime.now().isoformat(),
        "log_type": "status",
        "content": json.dumps({
            "engine_temperature": 85.5,
            "fuel_level": 75.0,
            "speed": 12.5,
            "pressure": 2.1,
            "vibration": 0.3
        }),
        "severity": "normal"
    }

@pytest.fixture
def sample_anomalous_log():
    """Fixture for an anomalous log entry"""
    return {
        "ship_id": "VSL002",
        "timestamp": datetime.now().isoformat(),
        "log_type": "warning",
        "content": json.dumps({
            "engine_temperature": 150.0,  # Abnormally high
            "fuel_level": 10.0,          # Very low
            "speed": 25.0,               # Higher than normal
            "pressure": 5.0,             # Abnormal pressure
            "vibration": 2.0             # High vibration
        }),
        "severity": "high"
    }

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "version" in response.json()
    assert "model_status" in response.json()

def test_create_log(sample_log):
    """Test log creation endpoint"""
    response = client.post("/api/logs", json=sample_log)
    assert response.status_code == 200
    data = response.json()
    assert data["ship_id"] == sample_log["ship_id"]
    assert "anomaly_score" in data

def test_create_anomalous_log(sample_anomalous_log):
    """Test log creation with anomalous data"""
    response = client.post("/api/logs", json=sample_anomalous_log)
    assert response.status_code == 200
    data = response.json()
    assert data["ship_id"] == sample_anomalous_log["ship_id"]
    assert data.get("is_anomaly", False)  # Should be detected as anomaly
    assert "analysis" in data  # Should include detailed analysis

def test_get_logs():
    """Test logs retrieval endpoint"""
    response = client.get("/api/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_logs_with_ship_id():
    """Test logs retrieval with ship_id filter"""
    ship_id = "VSL001"
    response = client.get(f"/api/logs?ship_id={ship_id}")
    assert response.status_code == 200
    logs = response.json()
    assert all(log["ship_id"] == ship_id for log in logs)

def test_get_anomalies_only():
    """Test logs retrieval with anomalies_only filter"""
    response = client.get("/api/logs?anomalies_only=true")
    assert response.status_code == 200
    logs = response.json()
    assert all(log.get("is_anomaly", False) for log in logs)

def test_get_alerts():
    """Test alerts retrieval endpoint"""
    response = client.get("/api/alerts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_alerts_with_ship_id():
    """Test alerts retrieval with ship_id filter"""
    ship_id = "VSL001"
    response = client.get(f"/api/alerts?ship_id={ship_id}")
    assert response.status_code == 200
    alerts = response.json()
    assert all(alert["ship_id"] == ship_id for alert in alerts)

def test_model_health():
    """Test model health check endpoint"""
    response = client.get("/api/model/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_analyze_log():
    """Test log analysis endpoint"""
    # First create a log
    response = client.post("/api/logs", json=sample_anomalous_log())
    assert response.status_code == 200
    log_data = response.json()
    
    # Then analyze it
    if "id" in log_data:
        response = client.post(f"/api/analyze/{log_data['id']}")
        assert response.status_code == 200
        analysis = response.json()
        assert "contributing_factors" in analysis
        assert "recommendations" in analysis

def test_train_models():
    """Test model training endpoint"""
    # Prepare training data
    training_data = {
        "logs": [
            sample_log(),
            sample_anomalous_log(),
            # Add more varied logs for better training
            {
                "ship_id": "VSL003",
                "timestamp": datetime.now().isoformat(),
                "log_type": "status",
                "content": json.dumps({
                    "engine_temperature": 90.0,
                    "fuel_level": 60.0,
                    "speed": 15.0,
                    "pressure": 2.5,
                    "vibration": 0.4
                }),
                "severity": "normal"
            }
        ]
    }
    
    response = client.post("/api/train", json=training_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "timestamp" in data
    assert "metrics" in data

def test_error_handling():
    """Test error handling"""
    # Test invalid log format
    invalid_log = {
        "ship_id": "VSL001"  # Missing required fields
    }
    response = client.post("/api/logs", json=invalid_log)
    assert response.status_code == 422  # Validation error
    
    # Test invalid ship_id format
    response = client.get("/api/logs?ship_id=")
    assert response.status_code == 200
    assert len(response.json()) == 0
    
    # Test non-existent log analysis
    response = client.post("/api/analyze/non_existent_id")
    assert response.status_code == 404

if __name__ == "__main__":
    pytest.main([__file__])
