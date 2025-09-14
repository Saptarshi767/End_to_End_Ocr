"""
Tesseract OCR engine implementation with enhanced confidence scoring and error handling.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import cv2
import os
import subprocess
import sys
from pathlib import Path

from .base_engine import BaseOCREngine
from ..core.models import OCRResult, BoundingBox, WordData, TableRegion
from ..core.exceptions import OCREngineError

logger = logging.getLogger(__name__)

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract dependencies not available. Install with: pip install pytesseract pillow")


class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR engine implementation.
    """
    
    def __init__(self, confidence_threshold: float = 0.8):
        super().__init__("tesseract", confidence_threshold)
        
        # Tesseract-specific configuration
        self.psm = 6  # Page segmentation mode
        self.oem = 3  # OCR Engine Mode
        self.tesseract_config = ""
        
        # Extended language support for Tesseract
        self.supported_languages = [
            'eng', 'fra', 'deu', 'spa', 'ita', 'por', 'rus', 'chi_sim', 'chi_tra', 'jpn', 'kor'
        ]
        
    def initialize(self) -> None:
        """Initialize Tesseract OCR engine with comprehensive validation."""
        if not TESSERACT_AVAILABLE:
            raise OCREngineError(
                "Tesseract dependencies not available. Install with: pip install pytesseract pillow",
                error_code="TESSERACT_DEPS_MISSING"
            )
            
        try:
            # Validate Tesseract installation
            self._validate_tesseract_installation()
            
            # Test Tesseract functionality
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
            # Validate available languages
            self._validate_languages()
            
            # Build configuration string
            self._build_config()
            
            # Test with a simple image
            self._test_basic_functionality()
            
            super().initialize()
            
        except OCREngineError:
            raise
        except Exception as e:
            raise OCREngineError(
                f"Failed to initialize Tesseract: {str(e)}", 
                error_code="TESSERACT_INIT_FAILED",
                context={"original_error": str(e)}
            )
            
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure Tesseract-specific parameters."""
        super().configure(config)
        
        # Tesseract-specific parameters
        if 'psm' in config:
            self.psm = config['psm']
            
        if 'oem' in config:
            self.oem = config['oem']
            
        if 'preserve_interword_spaces' in config:
            self.preserve_interword_spaces = config['preserve_interword_spaces']
            
        # Rebuild configuration string
        self._build_config()
        
    def _validate_tesseract_installation(self) -> None:
        """Validate that Tesseract is properly installed and accessible."""
        try:
            # Try to get Tesseract version
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise OCREngineError(
                "Tesseract executable not found. Please install Tesseract OCR and ensure it's in your PATH.",
                error_code="TESSERACT_NOT_FOUND"
            )
        except Exception as e:
            raise OCREngineError(
                f"Failed to validate Tesseract installation: {str(e)}",
                error_code="TESSERACT_VALIDATION_FAILED"
            )
    
    def _validate_languages(self) -> None:
        """Validate that required languages are available."""
        try:
            available_langs = pytesseract.get_languages(config='')
            missing_langs = [lang for lang in self.supported_languages if lang not in available_langs]
            
            if missing_langs:
                logger.warning(f"Some languages not available in Tesseract: {missing_langs}")
                # Filter out missing languages
                self.supported_languages = [lang for lang in self.supported_languages if lang in available_langs]
                
            if not self.supported_languages:
                raise OCREngineError(
                    "No supported languages available in Tesseract installation",
                    error_code="NO_LANGUAGES_AVAILABLE"
                )
                
            logger.info(f"Available Tesseract languages: {self.supported_languages}")
            
        except Exception as e:
            logger.warning(f"Could not validate languages: {str(e)}")
            # Continue with default languages
    
    def _test_basic_functionality(self) -> None:
        """Test basic Tesseract functionality with a simple image."""
        try:
            # Create a simple test image with text
            test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(test_image)
            
            # Test OCR
            text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            
            if not text or "TEST" not in text.upper():
                logger.warning("Tesseract basic functionality test did not return expected result")
            else:
                logger.debug("Tesseract basic functionality test passed")
                
        except Exception as e:
            logger.warning(f"Tesseract basic functionality test failed: {str(e)}")
    
    def _build_config(self) -> None:
        """Build Tesseract configuration string."""
        config_parts = [
            f"--psm {self.psm}",
            f"--oem {self.oem}"
        ]
        
        # Add additional configuration from config dict
        if 'preserve_interword_spaces' in self.config:
            config_parts.append(f"-c preserve_interword_spaces={self.config['preserve_interword_spaces']}")
            
        if 'tesseract_char_whitelist' in self.config:
            config_parts.append(f"-c tessedit_char_whitelist={self.config['tesseract_char_whitelist']}")
            
        if 'tesseract_char_blacklist' in self.config:
            config_parts.append(f"-c tessedit_char_blacklist={self.config['tesseract_char_blacklist']}")
            
        self.tesseract_config = " ".join(config_parts)
        logger.debug(f"Tesseract config: {self.tesseract_config}")
        
    def _extract_text_impl(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Extract text using Tesseract OCR with enhanced confidence scoring and error handling.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional parameters (language, etc.)
            
        Returns:
            OCR result with extracted text and metadata
        """
        try:
            # Validate input image
            self._validate_input_image(image)
            
            # Convert numpy array to PIL Image
            pil_image = self._convert_to_pil_image(image)
                
            # Get and validate language parameter
            language = self._validate_language(kwargs.get('language', 'eng'))
                
            # Extract detailed OCR data
            ocr_data = self._extract_ocr_data(pil_image, language)
            
            # Extract plain text
            text = self._extract_plain_text(pil_image, language)
            
            # Process word-level data with enhanced confidence scoring
            word_data, confidences = self._process_word_data(ocr_data)
            
            # Calculate enhanced overall confidence
            overall_confidence = self._calculate_overall_confidence(confidences, text, word_data)
            
            # Create optimized line-level bounding boxes
            line_boxes = self._group_words_into_lines(word_data)
            
            # Validate result quality
            result = OCRResult(
                text=text,
                confidence=overall_confidence,
                bounding_boxes=line_boxes,
                word_level_data=word_data
            )
            
            self._validate_ocr_result(result)
            
            return result
            
        except OCREngineError:
            raise
        except pytesseract.TesseractError as e:
            raise OCREngineError(
                f"Tesseract processing error: {str(e)}", 
                error_code="TESSERACT_PROCESSING_ERROR",
                context={"tesseract_error": str(e)}
            )
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {str(e)}")
            raise OCREngineError(
                f"Tesseract OCR failed: {str(e)}", 
                error_code="TESSERACT_EXTRACTION_FAILED",
                context={"original_error": str(e)}
            )
    
    def _validate_input_image(self, image: np.ndarray) -> None:
        """Validate input image format and properties."""
        if image is None or image.size == 0:
            raise OCREngineError("Input image is empty or None", error_code="INVALID_IMAGE")
            
        if len(image.shape) not in [2, 3]:
            raise OCREngineError(
                f"Invalid image shape: {image.shape}. Expected 2D or 3D array.", 
                error_code="INVALID_IMAGE_SHAPE"
            )
            
        # Check image dimensions
        height, width = image.shape[:2]
        if height < 10 or width < 10:
            raise OCREngineError(
                f"Image too small: {width}x{height}. Minimum size is 10x10 pixels.", 
                error_code="IMAGE_TOO_SMALL"
            )
            
        if height > 10000 or width > 10000:
            logger.warning(f"Large image detected: {width}x{height}. This may affect performance.")
    
    def _convert_to_pil_image(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image with proper format handling."""
        try:
            if len(image.shape) == 3:
                # Convert BGR to RGB if needed
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                return Image.fromarray(image)
            else:
                # Grayscale image
                return Image.fromarray(image, mode='L')
        except Exception as e:
            raise OCREngineError(
                f"Failed to convert image format: {str(e)}", 
                error_code="IMAGE_CONVERSION_FAILED"
            )
    
    def _validate_language(self, language: str) -> str:
        """Validate and return appropriate language code."""
        if not self.supports_language(language):
            logger.warning(f"Language {language} not supported, using 'eng'")
            if not self.supports_language('eng'):
                # Fallback to first available language
                if self.supported_languages:
                    fallback = self.supported_languages[0]
                    logger.warning(f"English not available, using {fallback}")
                    return fallback
                else:
                    raise OCREngineError(
                        "No supported languages available", 
                        error_code="NO_LANGUAGES_AVAILABLE"
                    )
            return 'eng'
        return language
    
    def _extract_ocr_data(self, pil_image: Image.Image, language: str) -> Dict[str, List]:
        """Extract detailed OCR data from image."""
        try:
            return pytesseract.image_to_data(
                pil_image,
                lang=language,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractError as e:
            raise OCREngineError(
                f"Tesseract data extraction failed: {str(e)}", 
                error_code="TESSERACT_DATA_EXTRACTION_FAILED"
            )
    
    def _extract_plain_text(self, pil_image: Image.Image, language: str) -> str:
        """Extract plain text from image."""
        try:
            return pytesseract.image_to_string(
                pil_image,
                lang=language,
                config=self.tesseract_config
            ).strip()
        except pytesseract.TesseractError as e:
            raise OCREngineError(
                f"Tesseract text extraction failed: {str(e)}", 
                error_code="TESSERACT_TEXT_EXTRACTION_FAILED"
            )
    
    def _process_word_data(self, data: Dict[str, List]) -> Tuple[List[WordData], List[float]]:
        """Process word-level data with enhanced confidence scoring."""
        word_data = []
        confidences = []
        
        for i in range(len(data['text'])):
            word_text = data['text'][i].strip()
            if not word_text:
                continue
                
            # Enhanced confidence calculation
            raw_confidence = float(data['conf'][i])
            confidence = self._calculate_word_confidence(raw_confidence, word_text, data, i)
            
            confidences.append(confidence)
            
            # Create bounding box with validation
            bbox = self._create_validated_bounding_box(data, i, confidence)
            
            # Create word data
            word_data.append(WordData(
                text=word_text,
                confidence=confidence,
                bounding_box=bbox
            ))
                
        return word_data, confidences
    
    def _calculate_word_confidence(self, raw_confidence: float, word_text: str, 
                                 data: Dict[str, List], index: int) -> float:
        """Calculate enhanced confidence score for a word."""
        # Convert Tesseract confidence to 0-1 range
        if raw_confidence < 0:  # Tesseract returns -1 for no confidence
            base_confidence = 0.0
        else:
            base_confidence = raw_confidence / 100.0
            
        # Apply confidence adjustments based on word characteristics
        confidence_multiplier = 1.0
        
        # Penalize very short words (likely noise)
        if len(word_text) == 1:
            confidence_multiplier *= 0.8
        elif len(word_text) == 2:
            confidence_multiplier *= 0.9
            
        # Boost confidence for dictionary words (simple heuristic)
        if self._is_likely_word(word_text):
            confidence_multiplier *= 1.1
            
        # Penalize words with unusual character patterns
        if self._has_unusual_pattern(word_text):
            confidence_multiplier *= 0.9
            
        # Consider bounding box dimensions if available
        if 'width' in data and 'height' in data and index < len(data['width']):
            width = data['width'][index]
            height = data['height'][index]
            if width > 0 and height > 0:
                aspect_ratio = width / height
                # Penalize extremely wide or tall bounding boxes
                if aspect_ratio > 10 or aspect_ratio < 0.1:
                    confidence_multiplier *= 0.8
                
        final_confidence = min(1.0, base_confidence * confidence_multiplier)
        return max(0.0, final_confidence)
    
    def _is_likely_word(self, text: str) -> bool:
        """Simple heuristic to check if text looks like a real word."""
        # Check for reasonable character patterns
        if len(text) < 2:
            return False
            
        # Check for reasonable vowel/consonant distribution
        vowels = set('aeiouAEIOU')
        has_vowel = any(c in vowels for c in text)
        has_consonant = any(c.isalpha() and c not in vowels for c in text)
        
        return has_vowel and has_consonant
    
    def _has_unusual_pattern(self, text: str) -> bool:
        """Check if text has unusual character patterns that might indicate OCR errors."""
        # Check for excessive special characters
        special_chars = sum(1 for c in text if not c.isalnum() and c not in ' -.,')
        if len(text) > 0 and special_chars / len(text) > 0.3:
            return True
            
        # Check for alternating case patterns (likely OCR errors)
        if len(text) > 3:
            case_changes = sum(1 for i in range(1, len(text)) 
                             if text[i].isalpha() and text[i-1].isalpha() 
                             and text[i].isupper() != text[i-1].isupper())
            if case_changes > len(text) // 2:
                return True
                
        return False
    
    def _create_validated_bounding_box(self, data: Dict[str, List], index: int, 
                                     confidence: float) -> BoundingBox:
        """Create a validated bounding box."""
        x = max(0, data['left'][index])
        y = max(0, data['top'][index])
        width = max(1, data['width'][index])
        height = max(1, data['height'][index])
        
        return BoundingBox(
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=confidence
        )
    
    def _calculate_overall_confidence(self, confidences: List[float], text: str, 
                                    word_data: List[WordData]) -> float:
        """Calculate enhanced overall confidence score."""
        if not confidences:
            return 0.0
            
        # Base confidence from word confidences
        base_confidence = np.mean(confidences)
        
        # Apply text-level adjustments
        confidence_multiplier = 1.0
        
        # Boost confidence if we have reasonable amount of text
        if len(text.strip()) > 10:
            confidence_multiplier *= 1.05
        elif len(text.strip()) < 3:
            confidence_multiplier *= 0.8
            
        # Consider word count
        if len(word_data) > 5:
            confidence_multiplier *= 1.02
        elif len(word_data) == 1:
            confidence_multiplier *= 0.9
            
        # Consider confidence distribution
        if confidences:
            confidence_std = np.std(confidences)
            if confidence_std < 0.1:  # Consistent confidence scores
                confidence_multiplier *= 1.05
            elif confidence_std > 0.3:  # Highly variable confidence
                confidence_multiplier *= 0.95
                
        final_confidence = min(1.0, base_confidence * confidence_multiplier)
        return max(0.0, final_confidence)
    
    def _validate_ocr_result(self, result: OCRResult) -> None:
        """Validate OCR result quality and log warnings if needed."""
        if not result.text or not result.text.strip():
            logger.warning("OCR result contains no text")
            
        if result.confidence < 0.3:
            logger.warning(f"Very low OCR confidence: {result.confidence:.2f}")
            
        if not result.word_level_data:
            logger.warning("No word-level data extracted")
            
        # Check for potential OCR errors
        if result.text and len(result.text) > 100:
            # Check for excessive special characters (might indicate OCR errors)
            special_char_ratio = sum(1 for c in result.text if not c.isalnum() and c not in ' \n\t.,!?-') / len(result.text)
            if special_char_ratio > 0.2:
                logger.warning(f"High special character ratio: {special_char_ratio:.2f} - possible OCR errors")
            
    def _group_words_into_lines(self, word_data: List[WordData]) -> List[BoundingBox]:
        """Group words into line-level bounding boxes."""
        if not word_data:
            return []
            
        lines = []
        current_line = []
        current_y = None
        line_height_threshold = 10  # Pixels
        
        for word in word_data:
            word_y = word.bounding_box.y
            
            if current_y is None or abs(word_y - current_y) <= line_height_threshold:
                # Same line
                current_line.append(word)
                current_y = word_y
            else:
                # New line
                if current_line:
                    lines.append(self._create_line_bbox(current_line))
                current_line = [word]
                current_y = word_y
                
        # Add last line
        if current_line:
            lines.append(self._create_line_bbox(current_line))
            
        return lines
        
    def _create_line_bbox(self, words: List[WordData]) -> BoundingBox:
        """Create a bounding box that encompasses all words in a line."""
        if not words:
            return BoundingBox(0, 0, 0, 0, 0.0)
            
        min_x = min(word.bounding_box.x for word in words)
        max_x = max(word.bounding_box.x + word.bounding_box.width for word in words)
        min_y = min(word.bounding_box.y for word in words)
        max_y = max(word.bounding_box.y + word.bounding_box.height for word in words)
        
        avg_confidence = np.mean([word.confidence for word in words])
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=avg_confidence
        )
        
    def detect_tables(self, image: np.ndarray, **kwargs) -> List[TableRegion]:
        """
        Detect table regions using Tesseract's layout analysis with enhanced error handling.
        """
        try:
            # Validate input
            self._validate_input_image(image)
            
            # Convert numpy array to PIL Image
            pil_image = self._convert_to_pil_image(image)
                
            # Use PSM 6 for table detection
            table_config = f"--psm 6 --oem {self.oem}"
            
            # Get layout information
            try:
                data = pytesseract.image_to_data(
                    pil_image,
                    config=table_config,
                    output_type=pytesseract.Output.DICT
                )
            except pytesseract.TesseractError as e:
                logger.error(f"Tesseract layout analysis failed: {str(e)}")
                raise OCREngineError(
                    f"Table detection failed: {str(e)}", 
                    error_code="TESSERACT_TABLE_DETECTION_FAILED"
                )
            
            # Group text blocks that might be tables
            table_regions = self._identify_table_regions(data, image.shape[:2])
            
            logger.info(f"Detected {len(table_regions)} potential table regions")
            return table_regions
            
        except OCREngineError:
            raise
        except Exception as e:
            logger.error(f"Tesseract table detection failed: {str(e)}")
            raise OCREngineError(
                f"Table detection failed: {str(e)}", 
                error_code="TABLE_DETECTION_ERROR",
                context={"original_error": str(e)}
            )
            
    def _identify_table_regions(self, data: Dict[str, List], image_shape: Tuple[int, int]) -> List[TableRegion]:
        """
        Identify potential table regions from Tesseract layout data with enhanced algorithms.
        """
        table_regions = []
        
        try:
            # Filter and process text blocks
            blocks = self._extract_text_blocks(data)
            
            if len(blocks) < 4:  # Minimum for a small table
                logger.debug("Insufficient text blocks for table detection")
                return table_regions
            
            # Group blocks into potential table regions
            potential_tables = self._group_blocks_into_tables(blocks, image_shape)
            
            # Validate and create table regions
            for table_blocks in potential_tables:
                table_region = self._create_table_region(table_blocks)
                if table_region:
                    table_regions.append(table_region)
                    
        except Exception as e:
            logger.error(f"Error identifying table regions: {str(e)}")
            
        return table_regions
    
    def _extract_text_blocks(self, data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Extract and filter text blocks from OCR data."""
        blocks = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            confidence = data['conf'][i]
            
            # Filter out low-confidence and empty blocks
            if not text or confidence < 30:  # Tesseract confidence threshold
                continue
                
            blocks.append({
                'text': text,
                'x': data['left'][i],
                'y': data['top'][i],
                'width': data['width'][i],
                'height': data['height'][i],
                'conf': confidence,
                'right': data['left'][i] + data['width'][i],
                'bottom': data['top'][i] + data['height'][i]
            })
            
        return blocks
    
    def _group_blocks_into_tables(self, blocks: List[Dict[str, Any]], 
                                image_shape: Tuple[int, int]) -> List[List[Dict[str, Any]]]:
        """Group text blocks into potential table regions using spatial analysis."""
        if not blocks:
            return []
            
        # Sort blocks by position
        blocks.sort(key=lambda b: (b['y'], b['x']))
        
        # Find rows by grouping blocks with similar y-coordinates
        rows = self._group_blocks_into_rows(blocks)
        
        # Find potential table regions
        table_groups = []
        
        if len(rows) >= 2:  # Need at least 2 rows for a table
            # Look for consecutive rows that might form a table
            current_table = []
            
            for i, row in enumerate(rows):
                if not current_table:
                    current_table = [row]
                else:
                    # Check if this row aligns with the current table
                    if self._rows_align(current_table[-1], row):
                        current_table.append(row)
                    else:
                        # End current table if it has enough rows
                        if len(current_table) >= 2:
                            table_blocks = [block for row in current_table for block in row]
                            table_groups.append(table_blocks)
                        current_table = [row]
            
            # Don't forget the last table
            if len(current_table) >= 2:
                table_blocks = [block for row in current_table for block in row]
                table_groups.append(table_blocks)
                
        return table_groups
    
    def _group_blocks_into_rows(self, blocks: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group text blocks into rows based on y-coordinate proximity."""
        if not blocks:
            return []
            
        rows = []
        current_row = [blocks[0]]
        row_height_threshold = 15  # Pixels
        
        for block in blocks[1:]:
            # Check if block belongs to current row
            current_row_y = np.mean([b['y'] for b in current_row])
            
            if abs(block['y'] - current_row_y) <= row_height_threshold:
                current_row.append(block)
            else:
                # Start new row
                if current_row:
                    # Sort current row by x-coordinate
                    current_row.sort(key=lambda b: b['x'])
                    rows.append(current_row)
                current_row = [block]
        
        # Add last row
        if current_row:
            current_row.sort(key=lambda b: b['x'])
            rows.append(current_row)
            
        return rows
    
    def _rows_align(self, row1: List[Dict[str, Any]], row2: List[Dict[str, Any]]) -> bool:
        """Check if two rows align well enough to be part of the same table."""
        if not row1 or not row2:
            return False
            
        # Check horizontal overlap
        row1_left = min(block['x'] for block in row1)
        row1_right = max(block['right'] for block in row1)
        row2_left = min(block['x'] for block in row2)
        row2_right = max(block['right'] for block in row2)
        
        # Calculate overlap
        overlap_left = max(row1_left, row2_left)
        overlap_right = min(row1_right, row2_right)
        overlap_width = max(0, overlap_right - overlap_left)
        
        row1_width = row1_right - row1_left
        row2_width = row2_right - row2_left
        min_width = min(row1_width, row2_width)
        
        # Require at least 50% overlap
        overlap_ratio = overlap_width / min_width if min_width > 0 else 0
        
        return overlap_ratio >= 0.5
    
    def _create_table_region(self, blocks: List[Dict[str, Any]]) -> Optional[TableRegion]:
        """Create a table region from a group of text blocks."""
        if not blocks:
            return None
            
        try:
            # Calculate bounding box
            min_x = min(block['x'] for block in blocks)
            max_x = max(block['right'] for block in blocks)
            min_y = min(block['y'] for block in blocks)
            max_y = max(block['bottom'] for block in blocks)
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Validate dimensions
            if width < 100 or height < 50:  # Minimum table size
                logger.debug(f"Table region too small: {width}x{height}")
                return None
                
            # Calculate average confidence
            avg_confidence = np.mean([block['conf'] for block in blocks]) / 100.0
            
            # Additional validation based on block distribution
            if not self._validate_table_structure(blocks):
                logger.debug("Table structure validation failed")
                return None
                
            return TableRegion(
                bounding_box=BoundingBox(
                    x=min_x,
                    y=min_y,
                    width=width,
                    height=height,
                    confidence=avg_confidence
                ),
                confidence=avg_confidence,
                page_number=1
            )
            
        except Exception as e:
            logger.error(f"Error creating table region: {str(e)}")
            return None
    
    def _validate_table_structure(self, blocks: List[Dict[str, Any]]) -> bool:
        """Validate that blocks form a reasonable table structure."""
        if len(blocks) < 4:
            return False
            
        # Check for reasonable distribution of blocks
        rows = self._group_blocks_into_rows(blocks)
        
        if len(rows) < 2:
            return False
            
        # Check that rows have similar number of columns (with some tolerance)
        row_lengths = [len(row) for row in rows]
        max_length = max(row_lengths)
        min_length = min(row_lengths)
        
        # Allow some variation in row lengths
        if max_length > 0 and min_length / max_length < 0.5:
            return False
            
        return True
        
    def preprocess_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess image for better Tesseract OCR accuracy.
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
                
            # Apply preprocessing based on configuration
            processed = gray
            
            if self.config.get('noise_reduction', True):
                # Apply Gaussian blur to reduce noise
                processed = cv2.GaussianBlur(processed, (3, 3), 0)
                
            if self.config.get('contrast_enhancement', True):
                # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
                
            if self.config.get('deskew', True):
                # Basic deskewing (simplified implementation)
                processed = self._deskew_image(processed)
                
            # Resize if specified
            resize_factor = self.config.get('resize_factor', 1.0)
            if resize_factor != 1.0:
                height, width = processed.shape
                new_height = int(height * resize_factor)
                new_width = int(width * resize_factor)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
            return processed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return image  # Return original image if preprocessing fails
            
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Basic image deskewing using Hough line detection.
        """
        try:
            # Apply edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = theta - np.pi/2
                    angles.append(angle)
                    
                if angles:
                    avg_angle = np.mean(angles)
                    
                    # Only deskew if angle is significant
                    if abs(avg_angle) > 0.01:  # ~0.5 degrees
                        # Rotate image
                        height, width = image.shape
                        center = (width // 2, height // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(avg_angle), 1.0)
                        
                        # Calculate new dimensions
                        cos_angle = abs(rotation_matrix[0, 0])
                        sin_angle = abs(rotation_matrix[0, 1])
                        new_width = int((height * sin_angle) + (width * cos_angle))
                        new_height = int((height * cos_angle) + (width * sin_angle))
                        
                        # Adjust translation
                        rotation_matrix[0, 2] += (new_width / 2) - center[0]
                        rotation_matrix[1, 2] += (new_height / 2) - center[1]
                        
                        # Apply rotation
                        deskewed = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                        return deskewed
                        
            return image
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {str(e)}")
            return image