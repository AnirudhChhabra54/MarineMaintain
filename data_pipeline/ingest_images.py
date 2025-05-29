import pytesseract
from PIL import Image
import logging
from typing import List, Dict, Any
import os
import re
from datetime import datetime
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageIngestionPipeline:
    def __init__(self):
        # Regular expressions for extracting information
        self.ship_id_pattern = r'Ship ID:\s*([A-Z0-9]+)'
        self.timestamp_pattern = r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})'
        self.log_type_pattern = r'Log Type:\s*(\w+)'
        
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding to get black and white image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Noise removal
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Dilation to make text more prominent
            kernel = np.ones((1, 1), np.uint8)
            dilated = cv2.dilate(denoised, kernel, iterations=1)
            
            return dilated

        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return image

    def extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")

            # Check file format
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {file_ext}")

            # Read image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Failed to load image")

            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Perform OCR
            text = pytesseract.image_to_string(processed_image)
            
            return text.strip()

        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""

    def parse_log_entry(self, text: str) -> Dict[str, Any]:
        """Parse extracted text to structured log entry."""
        try:
            # Extract information using regex patterns
            ship_id_match = re.search(self.ship_id_pattern, text)
            timestamp_match = re.search(self.timestamp_pattern, text)
            log_type_match = re.search(self.log_type_pattern, text)

            if not all([ship_id_match, timestamp_match]):
                logger.warning("Could not extract all required fields from text")
                return {}

            # Structure the extracted information
            log_entry = {
                'ship_id': ship_id_match.group(1) if ship_id_match else 'UNKNOWN',
                'timestamp': datetime.strptime(
                    timestamp_match.group(1),
                    '%Y-%m-%d %H:%M:%S'
                ).isoformat() if timestamp_match else datetime.now().isoformat(),
                'log_type': log_type_match.group(1) if log_type_match else 'manual',
                'content': text,
                'severity': 'normal',
                'anomaly_score': 0.0
            }

            return log_entry

        except Exception as e:
            logger.error(f"Error parsing log entry: {str(e)}")
            return {}

    def process_image(self, file_path: str) -> List[Dict[str, Any]]:
        """Process image file and return structured data."""
        try:
            # Extract text from image
            text = self.extract_text_from_image(file_path)
            if not text:
                return []

            # Parse the extracted text
            log_entry = self.parse_log_entry(text)
            if not log_entry:
                return []

            logger.info(f"Successfully processed log entry from image: {file_path}")
            return [log_entry]

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return []

    def ingest_image(self, file_path: str) -> bool:
        """Main ingestion method."""
        try:
            processed_data = self.process_image(file_path)
            if not processed_data:
                return False

            # Here you would typically send the data to your API
            # For now, we'll just log success
            logger.info(f"Successfully ingested log from image: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest image: {str(e)}")
            return False

    def batch_process_directory(self, directory_path: str) -> bool:
        """Process all supported images in a directory."""
        try:
            if not os.path.exists(directory_path):
                raise FileNotFoundError(f"Directory not found: {directory_path}")

            success_count = 0
            failure_count = 0

            # Process each supported image in directory
            for filename in os.listdir(directory_path):
                if os.path.splitext(filename)[1].lower() in self.supported_formats:
                    file_path = os.path.join(directory_path, filename)
                    if self.ingest_image(file_path):
                        success_count += 1
                    else:
                        failure_count += 1

            logger.info(f"Batch processing complete. Successes: {success_count}, Failures: {failure_count}")
            return success_count > 0

        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    pipeline = ImageIngestionPipeline()
    
    # Process single image
    success = pipeline.ingest_image("sample_log.jpg")
    print(f"Single image ingestion {'successful' if success else 'failed'}")
    
    # Process directory of images
    success = pipeline.batch_process_directory("sample_logs")
    print(f"Batch processing {'successful' if success else 'failed'}")
