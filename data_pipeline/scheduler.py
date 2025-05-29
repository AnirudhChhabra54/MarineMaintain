from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
import os
from typing import List, Dict, Any
from datetime import datetime
from ingest_csv import CSVIngestionPipeline
from ingest_pdf import PDFIngestionPipeline
from ingest_images import ImageIngestionPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IngestionScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.csv_pipeline = CSVIngestionPipeline()
        self.pdf_pipeline = PDFIngestionPipeline()
        self.image_pipeline = ImageIngestionPipeline()
        
        # Directories to monitor
        self.watch_directories = {
            'csv': 'data/incoming/csv',
            'pdf': 'data/incoming/pdf',
            'images': 'data/incoming/images'
        }
        
        # Processed files tracking
        self.processed_files: Dict[str, List[str]] = {
            'csv': [],
            'pdf': [],
            'images': []
        }

    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        try:
            for directory in self.watch_directories.values():
                os.makedirs(directory, exist_ok=True)
            logger.info("Watch directories setup complete")
        except Exception as e:
            logger.error(f"Error setting up directories: {str(e)}")

    def process_csv_files(self):
        """Process new CSV files in the watch directory."""
        try:
            directory = self.watch_directories['csv']
            for filename in os.listdir(directory):
                if filename.endswith('.csv') and filename not in self.processed_files['csv']:
                    file_path = os.path.join(directory, filename)
                    logger.info(f"Processing CSV file: {filename}")
                    
                    if self.csv_pipeline.ingest_csv(file_path):
                        self.processed_files['csv'].append(filename)
                        # Move to processed directory or delete as needed
                        logger.info(f"Successfully processed CSV file: {filename}")
                    else:
                        logger.error(f"Failed to process CSV file: {filename}")
        except Exception as e:
            logger.error(f"Error in CSV processing job: {str(e)}")

    def process_pdf_files(self):
        """Process new PDF files in the watch directory."""
        try:
            directory = self.watch_directories['pdf']
            for filename in os.listdir(directory):
                if filename.endswith('.pdf') and filename not in self.processed_files['pdf']:
                    file_path = os.path.join(directory, filename)
                    logger.info(f"Processing PDF file: {filename}")
                    
                    if self.pdf_pipeline.ingest_pdf(file_path):
                        self.processed_files['pdf'].append(filename)
                        # Move to processed directory or delete as needed
                        logger.info(f"Successfully processed PDF file: {filename}")
                    else:
                        logger.error(f"Failed to process PDF file: {filename}")
        except Exception as e:
            logger.error(f"Error in PDF processing job: {str(e)}")

    def process_image_files(self):
        """Process new image files in the watch directory."""
        try:
            directory = self.watch_directories['images']
            for filename in os.listdir(directory):
                if any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']):
                    if filename not in self.processed_files['images']:
                        file_path = os.path.join(directory, filename)
                        logger.info(f"Processing image file: {filename}")
                        
                        if self.image_pipeline.ingest_image(file_path):
                            self.processed_files['images'].append(filename)
                            # Move to processed directory or delete as needed
                            logger.info(f"Successfully processed image file: {filename}")
                        else:
                            logger.error(f"Failed to process image file: {filename}")
        except Exception as e:
            logger.error(f"Error in image processing job: {str(e)}")

    def cleanup_processed_files(self):
        """Clean up processed files list periodically to prevent memory growth."""
        try:
            current_time = datetime.now()
            # Clean up files processed more than 24 hours ago
            # In a production environment, you might want to store this in a database
            for file_type in self.processed_files:
                self.processed_files[file_type] = []
            logger.info("Cleaned up processed files tracking")
        except Exception as e:
            logger.error(f"Error in cleanup job: {str(e)}")

    def start(self):
        """Start the scheduler with configured jobs."""
        try:
            # Setup watch directories
            self.setup_directories()

            # Add jobs to the scheduler
            # Process CSV files every 5 minutes
            self.scheduler.add_job(
                self.process_csv_files,
                CronTrigger(minute='*/5'),
                id='csv_processing'
            )

            # Process PDF files every 10 minutes
            self.scheduler.add_job(
                self.process_pdf_files,
                CronTrigger(minute='*/10'),
                id='pdf_processing'
            )

            # Process image files every 15 minutes
            self.scheduler.add_job(
                self.process_image_files,
                CronTrigger(minute='*/15'),
                id='image_processing'
            )

            # Clean up processed files list daily
            self.scheduler.add_job(
                self.cleanup_processed_files,
                CronTrigger(hour=0),  # Run at midnight
                id='cleanup'
            )

            # Start the scheduler
            self.scheduler.start()
            logger.info("Ingestion scheduler started successfully")

        except Exception as e:
            logger.error(f"Error starting scheduler: {str(e)}")
            raise

    def stop(self):
        """Stop the scheduler."""
        try:
            self.scheduler.shutdown()
            logger.info("Ingestion scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {str(e)}")

if __name__ == "__main__":
    # Example usage
    scheduler = IngestionScheduler()
    try:
        scheduler.start()
        
        # Keep the script running
        try:
            while True:
                pass
        except KeyboardInterrupt:
            scheduler.stop()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        scheduler.stop()
