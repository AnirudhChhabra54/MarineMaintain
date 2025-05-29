# SeaLogix API Documentation

## 🔍 Overview

The SeaLogix API provides endpoints for log ingestion, anomaly detection, and maintenance management. This document details all available endpoints, their parameters, and example responses.

## 🌐 Base URL

```
http://localhost:8000
```

## 🔐 Authentication

Authentication will be required in production. Include the API key in the header:

```http
Authorization: Bearer your-api-key
```

## 📝 Endpoints

### Health Check

```http
GET /
```

Returns the API status and model information.

**Response**
```json
{
  "message": "Welcome to SeaLogix API",
  "version": "1.0.0",
  "model_status": "loaded"
}
```

### Create Log Entry

```http
POST /api/logs
```

Submit a new log entry for processing.

**Request Body**
```json
{
  "ship_id": "VSL001",
  "timestamp": "2024-01-20T10:30:00Z",
  "log_type": "status",
  "content": {
    "engine_temperature": 85.5,
    "fuel_level": 75.0,
    "speed": 12.5,
    "pressure": 2.1,
    "vibration": 0.3
  },
  "severity": "normal"
}
```

**Response**
```json
{
  "id": "log_123",
  "ship_id": "VSL001",
  "timestamp": "2024-01-20T10:30:00Z",
  "log_type": "status",
  "content": {
    "engine_temperature": 85.5,
    "fuel_level": 75.0,
    "speed": 12.5,
    "pressure": 2.1,
    "vibration": 0.3
  },
  "severity": "normal",
  "anomaly_score": 0.15,
  "is_anomaly": false,
  "detection_timestamp": "2024-01-20T10:30:01Z"
}
```

### Get Logs

```http
GET /api/logs
```

Retrieve log entries with optional filtering.

**Query Parameters**
- `ship_id` (optional): Filter by ship ID
- `anomalies_only` (optional): Set to "true" to show only anomalous logs

**Response**
```json
[
  {
    "id": "log_123",
    "ship_id": "VSL001",
    "timestamp": "2024-01-20T10:30:00Z",
    "log_type": "status",
    "content": {
      "engine_temperature": 85.5,
      "fuel_level": 75.0,
      "speed": 12.5
    },
    "severity": "normal",
    "anomaly_score": 0.15,
    "is_anomaly": false
  }
]
```

### Get Alerts

```http
GET /api/alerts
```

Retrieve system alerts.

**Query Parameters**
- `ship_id` (optional): Filter by ship ID

**Response**
```json
[
  {
    "id": "alert_456",
    "ship_id": "VSL001",
    "timestamp": "2024-01-20T10:35:00Z",
    "alert_type": "anomaly_detected",
    "message": "High engine temperature detected",
    "severity": "high"
  }
]
```

### Analyze Log

```http
POST /api/analyze/{log_id}
```

Perform detailed analysis of a specific log entry.

**Response**
```json
{
  "timestamp": "2024-01-20T10:35:01Z",
  "log_id": "log_123",
  "feature_importance": {
    "engine_temperature": 0.8,
    "fuel_level": 0.3,
    "speed": 0.2
  },
  "contributing_factors": [
    {
      "feature": "engine_temperature",
      "error": 0.85,
      "importance": 0.8
    }
  ],
  "recommendations": [
    "Check engine cooling system and perform temperature regulation (Severity: high)"
  ]
}
```

### Train Models

```http
POST /api/train
```

Train new ML models with provided logs.

**Request Body**
```json
{
  "logs": [
    {
      "ship_id": "VSL001",
      "timestamp": "2024-01-20T10:30:00Z",
      "log_type": "status",
      "content": {
        "engine_temperature": 85.5,
        "fuel_level": 75.0,
        "speed": 12.5
      },
      "severity": "normal"
    }
  ]
}
```

**Response**
```json
{
  "status": "success",
  "timestamp": "20240120_103000",
  "metrics": {
    "autoencoder": {
      "train_losses": [0.5, 0.3, 0.2],
      "val_losses": [0.6, 0.4, 0.3]
    },
    "isolation_forest": {
      "feature_importance": {
        "engine_temperature": 0.8,
        "fuel_level": 0.3,
        "speed": 0.2
      }
    }
  }
}
```

### Get Model Health

```http
GET /api/model/health
```

Check the health status of ML models.

**Response**
```json
{
  "status": "healthy",
  "last_checked": "2024-01-20T10:35:01Z",
  "models": {
    "preprocessor": {
      "loaded": true,
      "features": 15
    },
    "autoencoder": {
      "loaded": true,
      "threshold": 0.8
    },
    "isolation_forest": {
      "loaded": true,
      "threshold": -0.5
    }
  },
  "weights": {
    "autoencoder": 0.6,
    "isolation_forest": 0.4
  }
}
```

## 📊 Status Codes

- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 404: Not Found
- 422: Validation Error
- 500: Internal Server Error

## 🔄 Rate Limiting

The API implements rate limiting to ensure system stability:

- 100 requests per minute per IP
- 1000 requests per hour per API key

## 📝 Error Handling

All errors follow this format:

```json
{
  "detail": "Error message here",
  "status_code": 400,
  "timestamp": "2024-01-20T10:35:01Z"
}
```

## 🔒 Security

- All endpoints will require authentication in production
- Use HTTPS for all requests
- API keys should be kept secure
- Implement proper input validation
- Monitor for suspicious activity

## 📦 Data Formats

### Log Entry Schema
```json
{
  "ship_id": "string",
  "timestamp": "string (ISO 8601)",
  "log_type": "string",
  "content": "object",
  "severity": "string"
}
```

### Alert Schema
```json
{
  "ship_id": "string",
  "timestamp": "string (ISO 8601)",
  "alert_type": "string",
  "message": "string",
  "severity": "string"
}
```

## 🔄 Versioning

The API uses semantic versioning. The current version is v1.0.0.

## 📞 Support

For API support, contact:
- Email: support@sealogix.com
- Documentation: https://docs.sealogix.com
- Status Page: https://status.sealogix.com
