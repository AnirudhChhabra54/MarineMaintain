import PyPDF2
import logging
from typing import List, Dict, Any
from datetime import datetime
import re
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFIngestionPipeline:
    def __init__(self):
        # Regular expressions for extracting information
        self.ship_id_pattern = r'Ship ID:\s*([A-Z0-9]+)'
        self.timestamp_pattern = r'Timestamp:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
        self.log_type_pattern = r'Log Type:\s*(\w+)'

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")

            with open(file_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text

        except PyPDF2.PdfReadError as e:
            logger.error(f"Error reading PDF: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error processing PDF: {str(e)}")
            return ""

    def parse_log_entry(self, text: str) -> Dict[str, Any]:
        """Parse text to extract structured log information."""
        try:
            # Extract information using regex patterns
            ship_id_match = re.search(self.ship_id_pattern, text)
            timestamp_match = re.search(self.timestamp_pattern, text)
            log_type_match = re.search(self.log_type_pattern, text)

            if not all([ship_id_match, timestamp_match, log_type_match]):
                logger.warning("Could not extract all required fields from text")
                return {}

            # Structure the extracted information
            log_entry = {
                'ship_id': ship_id_match.group(1),
                'timestamp': datetime.strptime(
                    timestamp_match.group(1),
                    '%Y-%m-%d %H:%M:%S'
                ).isoformat(),
                'log_type': log_type_match.group(1),
                'content': text,
                'severity': 'normal',
                'anomaly_score': 0.0
            }

            return log_entry

        except Exception as e:
            logger.error(f"Error parsing log entry: {str(e)}")
            return {}

    def process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process PDF file and return structured data."""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(file_path)
            if not text:
                return []

            # Split text into log entries (assuming one entry per page)
            pages = text.split('\n\n')
            
            # Process each page
            processed_data = []
            for page in pages:
                if page.strip():
                    log_entry = self.parse_log_entry(page)
                    if log_entry:
                        processed_data.append(log_entry)

            logger.info(f"Successfully processed {len(processed_data)} log entries from PDF")
            return processed_data

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return []

    def ingest_pdf(self, file_path: str) -> bool:
        """Main ingestion method."""
        try:
            processed_data = self.process_pdf(file_path)
            if not processed_data:
                return False

            # Here you would typically send the data to your API
            # For now, we'll just log success
            logger.info(f"Successfully ingested {len(processed_data)} records from PDF")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest PDF: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    pipeline = PDFIngestionPipeline()
    success = pipeline.ingest_pdf("sample_logs.pdf")
    print(f"Ingestion {'successful' if success else 'failed'}")
