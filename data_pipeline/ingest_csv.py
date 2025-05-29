import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict, Any
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CSVIngestionPipeline:
    def __init__(self):
        self.required_columns = ['timestamp', 'ship_id', 'log_type', 'content']
        
    def validate_csv_structure(self, df: pd.DataFrame) -> bool:
        """Validate if CSV has required columns."""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        return True

    def process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file and return structured data."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Validate structure
            if not self.validate_csv_structure(df):
                raise ValueError("Invalid CSV structure")

            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Process and structure the data
            processed_data = []
            for _, row in df.iterrows():
                log_entry = {
                    'timestamp': row['timestamp'].isoformat(),
                    'ship_id': str(row['ship_id']),
                    'log_type': row['log_type'],
                    'content': row['content'],
                    'severity': row.get('severity', 'normal'),
                    'anomaly_score': float(row.get('anomaly_score', 0.0))
                }
                processed_data.append(log_entry)

            logger.info(f"Successfully processed {len(processed_data)} log entries")
            return processed_data

        except pd.errors.EmptyDataError:
            logger.error(f"Empty CSV file: {file_path}")
            return []
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error processing CSV: {str(e)}")
            return []

    def ingest_csv(self, file_path: str) -> bool:
        """Main ingestion method."""
        try:
            processed_data = self.process_csv(file_path)
            if not processed_data:
                return False

            # Here you would typically send the data to your API
            # For now, we'll just log success
            logger.info(f"Successfully ingested {len(processed_data)} records")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest CSV: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    pipeline = CSVIngestionPipeline()
    success = pipeline.ingest_csv("sample_logs.csv")
    print(f"Ingestion {'successful' if success else 'failed'}")
