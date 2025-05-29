import pytest
import os
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import json
from data_pipeline.ingest_csv import CSVIngestionPipeline
from data_pipeline.ingest_pdf import PDFIngestionPipeline
from data_pipeline.ingest_images import ImageIngestionPipeline
from data_pipeline.scheduler import IngestionScheduler

@pytest.fixture
def sample_data_dir(tmp_path):
    """Create a temporary directory with sample data files"""
    # Create directories
    data_dir = tmp_path / "data"
    csv_dir = data_dir / "incoming" / "csv"
    pdf_dir = data_dir / "incoming" / "pdf"
    img_dir = data_dir / "incoming" / "images"
    
    csv_dir.mkdir(parents=True)
    pdf_dir.mkdir(parents=True)
    img_dir.mkdir(parents=True)
    
    # Create sample CSV
    csv_data = pd.DataFrame({
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'ship_id': ['VSL001'],
        'log_type': ['status'],
        'content': ['{"engine_temperature": 85.5, "fuel_level": 75.0}']
    })
    csv_data.to_csv(csv_dir / "sample_log.csv", index=False)
    
    # Create sample PDF with text content
    with open(pdf_dir / "sample_log.pdf", "w") as f:
        f.write("Ship ID: VSL001\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Log Type: status\n")
        f.write("Engine temperature: 85.5\n")
        f.write("Fuel level: 75.0\n")
    
    # Create sample image with text
    img = Image.new('RGB', (800, 400), color='white')
    img.save(img_dir / "sample_log.jpg")
    
    return data_dir

class TestCSVIngestion:
    def test_csv_validation(self, sample_data_dir):
        pipeline = CSVIngestionPipeline()
        csv_path = sample_data_dir / "incoming" / "csv" / "sample_log.csv"
        
        # Test valid CSV
        assert pipeline.ingest_csv(str(csv_path)) == True
        
        # Test invalid CSV
        invalid_csv = pd.DataFrame({'invalid_column': [1, 2, 3]})
        invalid_path = sample_data_dir / "incoming" / "csv" / "invalid.csv"
        invalid_csv.to_csv(invalid_path, index=False)
        assert pipeline.ingest_csv(str(invalid_path)) == False

    def test_csv_processing(self, sample_data_dir):
        pipeline = CSVIngestionPipeline()
        csv_path = sample_data_dir / "incoming" / "csv" / "sample_log.csv"
        
        processed_data = pipeline.process_csv(str(csv_path))
        assert len(processed_data) > 0
        assert all(key in processed_data[0] for key in ['timestamp', 'ship_id', 'log_type', 'content'])

class TestPDFIngestion:
    def test_pdf_extraction(self, sample_data_dir):
        pipeline = PDFIngestionPipeline()
        pdf_path = sample_data_dir / "incoming" / "pdf" / "sample_log.pdf"
        
        # Test text extraction
        text = pipeline.extract_text_from_pdf(str(pdf_path))
        assert "Ship ID: VSL001" in text
        assert "Log Type: status" in text

    def test_pdf_processing(self, sample_data_dir):
        pipeline = PDFIngestionPipeline()
        pdf_path = sample_data_dir / "incoming" / "pdf" / "sample_log.pdf"
        
        # Test full processing
        processed_data = pipeline.process_pdf(str(pdf_path))
        assert len(processed_data) > 0
        first_entry = processed_data[0]
        assert first_entry.get('ship_id') == 'VSL001'
        assert first_entry.get('log_type') == 'status'

class TestImageIngestion:
    def test_image_preprocessing(self, sample_data_dir):
        pipeline = ImageIngestionPipeline()
        img_path = sample_data_dir / "incoming" / "images" / "sample_log.jpg"
        
        # Test image loading and preprocessing
        image = cv2.imread(str(img_path))
        processed_image = pipeline.preprocess_image(image)
        assert processed_image is not None
        assert processed_image.shape == image.shape

    def test_image_processing(self, sample_data_dir):
        pipeline = ImageIngestionPipeline()
        img_path = sample_data_dir / "incoming" / "images" / "sample_log.jpg"
        
        # Test full processing
        processed_data = pipeline.process_image(str(img_path))
        assert isinstance(processed_data, list)

    def test_batch_processing(self, sample_data_dir):
        pipeline = ImageIngestionPipeline()
        img_dir = sample_data_dir / "incoming" / "images"
        
        # Test batch processing
        success = pipeline.batch_process_directory(str(img_dir))
        assert success == True

class TestScheduler:
    def test_scheduler_initialization(self, sample_data_dir):
        scheduler = IngestionScheduler()
        assert scheduler.scheduler is not None
        
        # Test directory setup
        scheduler.setup_directories()
        for dir_type in scheduler.watch_directories.values():
            assert os.path.exists(dir_type)

    def test_csv_processing_job(self, sample_data_dir):
        scheduler = IngestionScheduler()
        scheduler.watch_directories['csv'] = str(sample_data_dir / "incoming" / "csv")
        
        # Test CSV processing
        scheduler.process_csv_files()
        assert len(scheduler.processed_files['csv']) > 0

    def test_pdf_processing_job(self, sample_data_dir):
        scheduler = IngestionScheduler()
        scheduler.watch_directories['pdf'] = str(sample_data_dir / "incoming" / "pdf")
        
        # Test PDF processing
        scheduler.process_pdf_files()
        assert len(scheduler.processed_files['pdf']) > 0

    def test_image_processing_job(self, sample_data_dir):
        scheduler = IngestionScheduler()
        scheduler.watch_directories['images'] = str(sample_data_dir / "incoming" / "images")
        
        # Test image processing
        scheduler.process_image_files()
        assert len(scheduler.processed_files['images']) > 0

    def test_cleanup_job(self, sample_data_dir):
        scheduler = IngestionScheduler()
        
        # Add some processed files
        scheduler.processed_files['csv'] = ['test1.csv', 'test2.csv']
        scheduler.processed_files['pdf'] = ['test1.pdf']
        scheduler.processed_files['images'] = ['test1.jpg', 'test2.jpg']
        
        # Test cleanup
        scheduler.cleanup_processed_files()
        assert len(scheduler.processed_files['csv']) == 0
        assert len(scheduler.processed_files['pdf']) == 0
        assert len(scheduler.processed_files['images']) == 0

    def test_scheduler_start_stop(self, sample_data_dir):
        scheduler = IngestionScheduler()
        
        # Test start
        scheduler.start()
        assert scheduler.scheduler.running
        
        # Test stop
        scheduler.stop()
        assert not scheduler.scheduler.running

if __name__ == "__main__":
    pytest.main([__file__])
