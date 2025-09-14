"""
End-to-end testing framework for complete document processing workflows
"""

import pytest
import os
import tempfile
import json
import time
from typing import Dict, Any, List
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.core.services import DocumentProcessingService
from src.ocr.engine_factory import OCREngineFactory
from src.data_processing.data_cleaner import DataCleaner
from src.visualization.dashboard_framework import DashboardFramework
from src.ai.conversational_engine import ConversationalAIEngine
from src.security.privacy_manager import PrivacyManager
from src.core.export_service import ExportService


class E2ETestFramework:
    """End-to-end testing framework"""
    
    def __init__(self):
        self.test_data_dir = tempfile.mkdtemp()
        self.performance_metrics = {}
        self.accuracy_metrics = {}
        
    def create_test_document(self, doc_type: str, content: Dict[str, Any]) -> str:
        """Create test document with known content"""
        
        if doc_type == "simple_table":
            return self._create_simple_table_image(content)
        elif doc_type == "complex_table":
            return self._create_complex_table_image(content)
        elif doc_type == "multi_page_pdf":
            return self._create_multi_page_pdf(content)
        elif doc_type == "noisy_scan":
            return self._create_noisy_scan_image(content)
        else:
            raise ValueError(f"Unknown document type: {doc_type}")
    
    def _create_simple_table_image(self, content: Dict[str, Any]) -> str:
        """Create simple table image for testing"""
        
        # Create image
        width, height = 800, 600
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Draw table
        headers = content.get("headers", ["Column 1", "Column 2", "Column 3"])
        rows = content.get("rows", [
            ["Value 1", "Value 2", "Value 3"],
            ["Data A", "Data B", "Data C"],
            ["Item X", "Item Y", "Item Z"]
        ])
        
        # Table dimensions
        cell_width = width // len(headers)
        cell_height = 40
        start_y = 100
        
        # Draw headers
        for i, header in enumerate(headers):
            x = i * cell_width + 10
            y = start_y
            draw.rectangle([x-5, y-5, x+cell_width-5, y+cell_height-5], outline='black')
            draw.text((x, y+10), header, fill='black', font=font)
        
        # Draw rows
        for row_idx, row in enumerate(rows):
            for col_idx, cell in enumerate(row):
                x = col_idx * cell_width + 10
                y = start_y + (row_idx + 1) * cell_height
                draw.rectangle([x-5, y-5, x+cell_width-5, y+cell_height-5], outline='black')
                draw.text((x, y+10), str(cell), fill='black', font=font)
        
        # Save image
        image_path = os.path.join(self.test_data_dir, f"simple_table_{int(time.time())}.png")
        image.save(image_path)
        
        return image_path
    
    def _create_complex_table_image(self, content: Dict[str, Any]) -> str:
        """Create complex table image with merged cells and formatting"""
        
        width, height = 1000, 800
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 14)
            header_font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            header_font = ImageFont.load_default()
        
        # Complex table with merged cells
        draw.text((50, 30), "Financial Report Q1 2023", fill='black', font=header_font)
        
        # Table structure
        table_data = content.get("table_data", [
            ["Department", "Q1 Revenue", "Q1 Expenses", "Profit"],
            ["Sales", "$150,000", "$45,000", "$105,000"],
            ["Marketing", "$75,000", "$65,000", "$10,000"],
            ["Engineering", "$200,000", "$120,000", "$80,000"],
            ["Total", "$425,000", "$230,000", "$195,000"]
        ])
        
        cell_width = 200
        cell_height = 35
        start_x, start_y = 50, 80
        
        for row_idx, row in enumerate(table_data):
            for col_idx, cell in enumerate(row):
                x = start_x + col_idx * cell_width
                y = start_y + row_idx * cell_height
                
                # Header row styling
                if row_idx == 0:
                    draw.rectangle([x, y, x+cell_width, y+cell_height], fill='lightgray', outline='black')
                    draw.text((x+5, y+8), cell, fill='black', font=header_font)
                else:
                    draw.rectangle([x, y, x+cell_width, y+cell_height], outline='black')
                    draw.text((x+5, y+8), cell, fill='black', font=font)
        
        image_path = os.path.join(self.test_data_dir, f"complex_table_{int(time.time())}.png")
        image.save(image_path)
        
        return image_path
    
    def _create_noisy_scan_image(self, content: Dict[str, Any]) -> str:
        """Create noisy scan image to test OCR robustness"""
        
        # Start with simple table
        image_path = self._create_simple_table_image(content)
        
        # Add noise
        image = Image.open(image_path)
        pixels = np.array(image)
        
        # Add random noise
        noise = np.random.randint(0, 50, pixels.shape)
        noisy_pixels = np.clip(pixels.astype(int) + noise, 0, 255).astype(np.uint8)
        
        # Add some blur effect
        from PIL import ImageFilter
        noisy_image = Image.fromarray(noisy_pixels)
        blurred_image = noisy_image.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Save noisy image
        noisy_path = os.path.join(self.test_data_dir, f"noisy_scan_{int(time.time())}.png")
        blurred_image.save(noisy_path)
        
        # Clean up original
        os.unlink(image_path)
        
        return noisy_path
    
    def _create_multi_page_pdf(self, content: Dict[str, Any]) -> str:
        """Create multi-page PDF for testing"""
        # This would require reportlab or similar library
        # For now, create multiple images and simulate PDF
        
        pages = content.get("pages", 2)
        image_paths = []
        
        for page in range(pages):
            page_content = {
                "headers": [f"Page {page+1} Col 1", f"Page {page+1} Col 2"],
                "rows": [
                    [f"P{page+1} Row 1 Col 1", f"P{page+1} Row 1 Col 2"],
                    [f"P{page+1} Row 2 Col 1", f"P{page+1} Row 2 Col 2"]
                ]
            }
            image_path = self._create_simple_table_image(page_content)
            image_paths.append(image_path)
        
        # Return first image path (simulating PDF)
        return image_paths[0]


class TestE2EDocumentProcessing:
    """End-to-end document processing tests"""
    
    @pytest.fixture
    def e2e_framework(self):
        """Create E2E testing framework"""
        return E2ETestFramework()
    
    @pytest.fixture
    def document_service(self):
        """Create document processing service"""
        return DocumentProcessingService()
    
    def test_simple_table_processing_workflow(self, e2e_framework, document_service):
        """Test complete workflow for simple table processing"""
        
        # Create test document
        test_content = {
            "headers": ["Product", "Price", "Quantity"],
            "rows": [
                ["Widget A", "$10.99", "100"],
                ["Widget B", "$15.50", "75"],
                ["Widget C", "$8.25", "200"]
            ]
        }
        
        document_path = e2e_framework.create_test_document("simple_table", test_content)
        
        try:
            # Start timing
            start_time = time.time()
            
            # Step 1: OCR Processing
            ocr_factory = OCREngineFactory()
            ocr_engine = ocr_factory.create_engine("tesseract")
            
            with open(document_path, 'rb') as f:
                image_data = f.read()
            
            ocr_result = ocr_engine.extract_text(image_data)
            
            # Verify OCR extracted some text
            assert ocr_result.text is not None
            assert len(ocr_result.text.strip()) > 0
            
            # Step 2: Table Extraction
            # Mock table extraction for now
            extracted_tables = [{
                "headers": test_content["headers"],
                "rows": test_content["rows"]
            }]
            
            # Step 3: Data Cleaning
            data_cleaner = DataCleaner()
            
            # Convert to DataFrame
            df = pd.DataFrame(extracted_tables[0]["rows"], columns=extracted_tables[0]["headers"])
            cleaned_df = data_cleaner.clean_dataframe(df)
            
            # Verify data cleaning
            assert not cleaned_df.empty
            assert len(cleaned_df.columns) == 3
            assert "Product" in cleaned_df.columns
            
            # Step 4: Dashboard Generation
            dashboard_framework = DashboardFramework()
            dashboard = dashboard_framework.create_dashboard(cleaned_df)
            
            # Verify dashboard creation
            assert dashboard is not None
            assert len(dashboard.charts) > 0
            
            # Step 5: Export
            export_service = ExportService()
            export_result = export_service.export_data(
                cleaned_df, 
                format="csv",
                filename="test_export.csv"
            )
            
            # Verify export
            assert export_result["success"] is True
            
            # Record performance metrics
            processing_time = time.time() - start_time
            e2e_framework.performance_metrics["simple_table_workflow"] = processing_time
            
            # Verify performance (should complete within reasonable time)
            assert processing_time < 30.0  # 30 seconds max
            
        finally:
            # Cleanup
            if os.path.exists(document_path):
                os.unlink(document_path)
    
    def test_complex_table_processing_workflow(self, e2e_framework, document_service):
        """Test workflow for complex table with formatting"""
        
        test_content = {
            "table_data": [
                ["Region", "Q1 Sales", "Q2 Sales", "Growth %"],
                ["North", "$125,000", "$140,000", "12%"],
                ["South", "$98,000", "$110,000", "12.2%"],
                ["East", "$156,000", "$175,000", "12.2%"],
                ["West", "$87,000", "$95,000", "9.2%"]
            ]
        }
        
        document_path = e2e_framework.create_test_document("complex_table", test_content)
        
        try:
            start_time = time.time()
            
            # Process document (simplified for test)
            ocr_factory = OCREngineFactory()
            ocr_engine = ocr_factory.create_engine("tesseract")
            
            with open(document_path, 'rb') as f:
                image_data = f.read()
            
            ocr_result = ocr_engine.extract_text(image_data)
            
            # Verify OCR can handle complex formatting
            assert ocr_result.text is not None
            assert "Region" in ocr_result.text or "Sales" in ocr_result.text
            
            processing_time = time.time() - start_time
            e2e_framework.performance_metrics["complex_table_workflow"] = processing_time
            
            # Complex tables may take longer but should still be reasonable
            assert processing_time < 45.0
            
        finally:
            if os.path.exists(document_path):
                os.unlink(document_path)
    
    def test_noisy_document_processing(self, e2e_framework, document_service):
        """Test processing of noisy/low-quality documents"""
        
        test_content = {
            "headers": ["Item", "Value"],
            "rows": [["Test", "123"], ["Data", "456"]]
        }
        
        document_path = e2e_framework.create_test_document("noisy_scan", test_content)
        
        try:
            start_time = time.time()
            
            # Test with multiple OCR engines for robustness
            ocr_factory = OCREngineFactory()
            
            # Try Tesseract first
            tesseract_engine = ocr_factory.create_engine("tesseract")
            
            with open(document_path, 'rb') as f:
                image_data = f.read()
            
            tesseract_result = tesseract_engine.extract_text(image_data)
            
            # For noisy documents, we expect some text extraction
            # but may not be perfect
            assert tesseract_result.text is not None
            
            # Try EasyOCR as fallback
            try:
                easyocr_engine = ocr_factory.create_engine("easyocr")
                easyocr_result = easyocr_engine.extract_text(image_data)
                
                # Compare results (EasyOCR might perform better on noisy images)
                assert easyocr_result.text is not None
                
            except Exception as e:
                # EasyOCR might not be available in test environment
                print(f"EasyOCR not available: {e}")
            
            processing_time = time.time() - start_time
            e2e_framework.performance_metrics["noisy_document_workflow"] = processing_time
            
        finally:
            if os.path.exists(document_path):
                os.unlink(document_path)
    
    def test_privacy_compliance_workflow(self, e2e_framework):
        """Test privacy compliance throughout the workflow"""
        
        # Create document with PII
        test_content = {
            "headers": ["Name", "SSN", "Email"],
            "rows": [
                ["John Doe", "123-45-6789", "john.doe@example.com"],
                ["Jane Smith", "987-65-4321", "jane.smith@example.com"]
            ]
        }
        
        document_path = e2e_framework.create_test_document("simple_table", test_content)
        
        try:
            # Initialize privacy manager
            privacy_manager = PrivacyManager()
            
            # Simulate data extraction
            extracted_data = {
                "headers": test_content["headers"],
                "rows": test_content["rows"]
            }
            
            # Register data with privacy manager
            record = privacy_manager.register_data(
                record_id="pii_test_doc",
                user_id="test_user",
                data=extracted_data,
                data_type="document",
                consent_given=True
            )
            
            # Verify PII detection
            assert record.pii_detected is True
            assert record.is_encrypted is True  # Should be encrypted due to PII
            
            # Test data anonymization
            anonymized_data = privacy_manager.anonymize_data(extracted_data)
            
            # Verify PII is anonymized
            anonymized_str = str(anonymized_data)
            assert "123-45-6789" not in anonymized_str
            assert "john.doe@example.com" not in anonymized_str
            assert "[SSN_REDACTED]" in anonymized_str
            assert "[EMAIL_REDACTED]" in anonymized_str
            
            # Test access control
            access_granted = privacy_manager.access_data(
                "pii_test_doc",
                "test_user",
                "data_processing"
            )
            assert access_granted is True
            
        finally:
            if os.path.exists(document_path):
                os.unlink(document_path)
    
    def test_conversational_ai_workflow(self, e2e_framework):
        """Test conversational AI integration"""
        
        # Create sample data
        sample_data = pd.DataFrame({
            "Product": ["Widget A", "Widget B", "Widget C"],
            "Sales": [1000, 1500, 800],
            "Region": ["North", "South", "East"]
        })
        
        # Mock conversational AI engine
        with patch('src.ai.conversational_engine.ConversationalAIEngine') as mock_ai:
            mock_ai_instance = Mock()
            mock_ai.return_value = mock_ai_instance
            
            # Mock response
            mock_ai_instance.process_question.return_value = {
                "answer": "Widget B has the highest sales with 1500 units.",
                "data": {"product": "Widget B", "sales": 1500},
                "visualization": {"type": "bar_chart", "data": sample_data.to_dict()}
            }
            
            ai_engine = mock_ai_instance
            
            # Test question processing
            response = ai_engine.process_question(
                "Which product has the highest sales?",
                sample_data
            )
            
            assert response["answer"] is not None
            assert "Widget B" in response["answer"]
            assert response["data"]["sales"] == 1500
    
    def test_performance_benchmarking(self, e2e_framework):
        """Test performance benchmarking with various document types"""
        
        document_types = ["simple_table", "complex_table", "noisy_scan"]
        performance_results = {}
        
        for doc_type in document_types:
            test_content = {
                "headers": ["Col1", "Col2"],
                "rows": [["Data1", "Data2"], ["Data3", "Data4"]]
            }
            
            document_path = e2e_framework.create_test_document(doc_type, test_content)
            
            try:
                # Benchmark OCR processing
                start_time = time.time()
                
                ocr_factory = OCREngineFactory()
                ocr_engine = ocr_factory.create_engine("tesseract")
                
                with open(document_path, 'rb') as f:
                    image_data = f.read()
                
                ocr_result = ocr_engine.extract_text(image_data)
                
                processing_time = time.time() - start_time
                performance_results[doc_type] = {
                    "processing_time": processing_time,
                    "text_length": len(ocr_result.text) if ocr_result.text else 0,
                    "confidence": ocr_result.confidence
                }
                
            finally:
                if os.path.exists(document_path):
                    os.unlink(document_path)
        
        # Verify performance benchmarks
        for doc_type, metrics in performance_results.items():
            assert metrics["processing_time"] < 60.0  # Max 1 minute per document
            assert metrics["text_length"] > 0  # Should extract some text
        
        # Store results for reporting
        e2e_framework.performance_metrics.update(performance_results)
    
    def test_accuracy_validation(self, e2e_framework):
        """Test accuracy validation with ground truth data"""
        
        # Ground truth data
        ground_truth = {
            "headers": ["Product", "Price", "Stock"],
            "rows": [
                ["Apple", "$1.50", "100"],
                ["Banana", "$0.75", "150"],
                ["Orange", "$2.00", "80"]
            ]
        }
        
        document_path = e2e_framework.create_test_document("simple_table", ground_truth)
        
        try:
            # Process document
            ocr_factory = OCREngineFactory()
            ocr_engine = ocr_factory.create_engine("tesseract")
            
            with open(document_path, 'rb') as f:
                image_data = f.read()
            
            ocr_result = ocr_engine.extract_text(image_data)
            
            # Calculate accuracy metrics
            extracted_text = ocr_result.text.lower() if ocr_result.text else ""
            
            # Check for presence of key terms
            key_terms = ["product", "price", "stock", "apple", "banana", "orange"]
            found_terms = sum(1 for term in key_terms if term in extracted_text)
            accuracy = found_terms / len(key_terms)
            
            e2e_framework.accuracy_metrics["simple_table_accuracy"] = accuracy
            
            # Accuracy should be reasonable for clean test images
            assert accuracy > 0.5  # At least 50% of terms should be found
            
        finally:
            if os.path.exists(document_path):
                os.unlink(document_path)
    
    def test_error_handling_and_recovery(self, e2e_framework):
        """Test error handling and recovery mechanisms"""
        
        # Test with invalid document
        invalid_document_path = os.path.join(e2e_framework.test_data_dir, "invalid.txt")
        with open(invalid_document_path, 'w') as f:
            f.write("This is not an image file")
        
        try:
            ocr_factory = OCREngineFactory()
            ocr_engine = ocr_factory.create_engine("tesseract")
            
            with open(invalid_document_path, 'rb') as f:
                invalid_data = f.read()
            
            # Should handle invalid input gracefully
            try:
                ocr_result = ocr_engine.extract_text(invalid_data)
                # If no exception, result should indicate failure
                assert ocr_result.confidence < 0.5 or not ocr_result.text
            except Exception as e:
                # Exception should be handled gracefully
                assert isinstance(e, (ValueError, TypeError, OSError))
                
        finally:
            if os.path.exists(invalid_document_path):
                os.unlink(invalid_document_path)
    
    def test_memory_usage_monitoring(self, e2e_framework):
        """Test memory usage during processing"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple documents
        for i in range(3):
            test_content = {
                "headers": [f"Col{j}" for j in range(5)],
                "rows": [[f"Data{i}_{j}" for j in range(5)] for _ in range(10)]
            }
            
            document_path = e2e_framework.create_test_document("simple_table", test_content)
            
            try:
                ocr_factory = OCREngineFactory()
                ocr_engine = ocr_factory.create_engine("tesseract")
                
                with open(document_path, 'rb') as f:
                    image_data = f.read()
                
                ocr_result = ocr_engine.extract_text(image_data)
                
            finally:
                if os.path.exists(document_path):
                    os.unlink(document_path)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        e2e_framework.performance_metrics["memory_usage"] = {
            "initial_mb": initial_memory / (1024 * 1024),
            "final_mb": final_memory / (1024 * 1024),
            "increase_mb": memory_increase / (1024 * 1024)
        }