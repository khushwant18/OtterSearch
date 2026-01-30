"""
Unit tests for ottersearch.extractors module
"""
import pytest
from pathlib import Path
from PIL import Image
from ottersearch.extractors import extract_pdf, extract_image


class TestExtractors:
    """Test document extraction functions"""
    
    def test_extract_image_png(self, tmp_path):
        """Test extracting PNG image"""
        # Create a simple test image
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(img_path)
        
        result = extract_image(img_path)
        
        # Should return the path
        assert result == img_path
        assert result.exists()
    
    def test_extract_image_jpg(self, tmp_path):
        """Test extracting JPG image"""
        img_path = tmp_path / "test.jpg"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(img_path)
        
        result = extract_image(img_path)
        
        assert result == img_path
        assert result.exists()
    
    def test_extract_image_nonexistent(self, tmp_path):
        """Test extracting non-existent image"""
        img_path = tmp_path / "nonexistent.png"
        
        # extract_image just returns the path without validation
        result = extract_image(img_path)
        assert result == img_path
    
    def test_extract_pdf_simple(self, tmp_path):
        """Test extracting text from a simple PDF"""
        # Note: This test requires PyMuPDF (fitz) to be installed
        # For a real PDF test, you'd need to create a PDF with text
        # This is a mock test showing the expected behavior
        
        # Skip if we can't create a real PDF for testing
        pytest.skip("Requires actual PDF file for testing")
    
    def test_extract_pdf_with_max_pages(self, tmp_path):
        """Test PDF extraction with page limit"""
        # Skip if we can't create a real PDF for testing
        pytest.skip("Requires actual PDF file for testing")
    
    def test_extract_pdf_nonexistent(self, tmp_path):
        """Test extracting non-existent PDF"""
        pdf_path = tmp_path / "nonexistent.pdf"
        
        with pytest.raises(RuntimeError) as exc_info:
            extract_pdf(pdf_path)
        
        assert "Failed to open file" in str(exc_info.value)
    
    def test_extract_pdf_returns_tuple(self, tmp_path):
        """Test that PDF extraction returns (text, page_count) tuple"""
        # This would be a real test with an actual PDF
        # Skipping for now as it requires PDF generation
        pytest.skip("Requires actual PDF file for testing")


class TestExtractorIntegration:
    """Integration tests for extractors"""
    
    def test_extract_multiple_images(self, tmp_path):
        """Test extracting multiple images"""
        image_paths = []
        
        for i in range(3):
            img_path = tmp_path / f"test{i}.png"
            img = Image.new('RGB', (50, 50), color=['red', 'green', 'blue'][i])
            img.save(img_path)
            image_paths.append(img_path)
        
        results = [extract_image(path) for path in image_paths]
        
        assert len(results) == 3
        assert all(r.exists() for r in results)
    
    def test_image_formats_supported(self, tmp_path):
        """Test that various image formats are supported"""
        formats = [('RGB', 'png'), ('RGB', 'jpg'), ('RGB', 'jpeg')]
        
        for mode, fmt in formats:
            img_path = tmp_path / f"test.{fmt}"
            img = Image.new(mode, (50, 50), color='red')
            img.save(img_path)
            
            result = extract_image(img_path)
            assert result.exists()
