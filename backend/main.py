from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
import uvicorn
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="SeaLogix API",
    description="AI-Driven Ship Maintenance System API",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ShipLog(BaseModel):
    ship_id: str
    timestamp: datetime
    log_type: str
    content: str
    severity: Optional[str] = "normal"
    anomaly_score: Optional[float] = 0.0

class Alert(BaseModel):
    ship_id: str
    timestamp: datetime
    alert_type: str
    message: str
    severity: str

# In-memory storage (replace with database in production)
ship_logs: List[ShipLog] = []
alerts: List[Alert] = []

@app.get("/")
async def root():
    return {"message": "Welcome to SeaLogix API"}

@app.post("/api/logs", response_model=ShipLog)
async def create_log(log: ShipLog):
    try:
        ship_logs.append(log)
        # Simulate anomaly detection
        if log.anomaly_score > 0.8:
            alert = Alert(
                ship_id=log.ship_id,
                timestamp=datetime.now(),
                alert_type="anomaly_detected",
                message=f"High anomaly score detected: {log.anomaly_score}",
                severity="high"
            )
            alerts.append(alert)
        return log
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/logs", response_model=List[ShipLog])
async def get_logs(ship_id: Optional[str] = None):
    try:
        if ship_id:
            return [log for log in ship_logs if log.ship_id == ship_id]
        return ship_logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts", response_model=List[Alert])
async def get_alerts(ship_id: Optional[str] = None):
    try:
        if ship_id:
            return [alert for alert in alerts if alert.ship_id == ship_id]
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
