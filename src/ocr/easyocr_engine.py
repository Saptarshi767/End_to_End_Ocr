"""
EasyOCR engine implementation with multi-language support and handwritten text recognition.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .base_engine import BaseOCREngine
from ..core.models import OCRResult, BoundingBox, WordData, TableRegion
from ..core.exceptions import OCREngineError

logger = logging.getLogger(__name__)

try:
    import easyocr
    import torch
    EASYOCR_AVAILABLE = True
    EasyOCRReader = easyocr.Reader
except ImportError:
    EASYOCR_AVAILABLE = False
    EasyOCRReader = Any  # Fallback type for when easyocr is not available
    logger.warning("EasyOCR not available. Install with: pip install easyocr torch")


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR engine implementation with multi-language support and handwritten text recognition.
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        super().__init__("easyocr", confidence_threshold)
        
        # EasyOCR reader instances (for different language combinations)
        self.readers = {}
        self.reader_lock = threading.Lock()
        
        # EasyOCR-specific configuration
        self.width_ths = 0.7
        self.height_ths = 0.7
        self.decoder = 'greedy'
        self.beamWidth = 5
        self.batch_size = 4  # Increased for better batch processing
        self.workers = 0
        self.allowlist = None
        self.blocklist = None
        
        # Handwriting recognition settings
        self.handwriting_enabled = True
        self.handwriting_threshold = 0.6  # Lower threshold for handwritten text
        
        # Batch processing settings
        self.max_batch_size = 8
        self.batch_timeout = 30  # seconds
        
        # Extended language support for EasyOCR with handwriting capabilities
        self.supported_languages = [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi', 'ar', 'ru',
            'fr', 'de', 'es', 'pt', 'it', 'nl', 'pl', 'tr', 'hi', 'bn',
            'ta', 'te', 'kn', 'ml', 'mr', 'ne', 'si', 'ur', 'fa', 'bg',
            'hr', 'cs', 'da', 'et', 'fi', 'hu', 'is', 'lv', 'lt', 'mt',
            'no', 'ro', 'sk', 'sl', 'sv', 'sw', 'tl', 'cy', 'eu', 'ga'
        ]
        
        # Languages with good handwriting support
        self.handwriting_languages = [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'ar', 'hi', 'bn', 'th'
        ]
        
        # Default languages to load
        self.active_languages = ['en']
        
        # Performance monitoring
        self.processing_stats = {
            'total_processed': 0,
            'batch_processed': 0,
            'avg_processing_time': 0.0,
            'handwriting_detected': 0
        }
        
    def initialize(self) -> None:
        """Initialize EasyOCR engine with enhanced capabilities."""
        if not EASYOCR_AVAILABLE:
            raise OCREngineError(
                "EasyOCR not available. Install with: pip install easyocr torch",
                error_code="EASYOCR_DEPS_MISSING"
            )
            
        try:
            # Check system capabilities
            gpu_available = self._check_gpu_availability()
            logger.info(f"GPU available for EasyOCR: {gpu_available}")
            
            # Initialize primary reader with default languages
            reader_key = tuple(sorted(self.active_languages))
            self.readers[reader_key] = easyocr.Reader(
                self.active_languages,
                gpu=gpu_available,
                verbose=False,
                download_enabled=True
            )
            
            # Test basic functionality
            self._test_basic_functionality()
            
            # Initialize handwriting detection if enabled
            if self.handwriting_enabled:
                self._initialize_handwriting_support()
            
            logger.info(f"EasyOCR initialized with languages: {self.active_languages}")
            logger.info(f"Handwriting support: {'enabled' if self.handwriting_enabled else 'disabled'}")
            logger.info(f"Batch processing: max_size={self.max_batch_size}")
            
            super().initialize()
            
        except Exception as e:
            raise OCREngineError(
                f"Failed to initialize EasyOCR: {str(e)}", 
                error_code="EASYOCR_INIT_FAILED",
                context={"original_error": str(e)}
            )
            
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure EasyOCR-specific parameters."""
        super().configure(config)
        
        # EasyOCR-specific parameters
        if 'width_ths' in config:
            self.width_ths = config['width_ths']
            
        if 'height_ths' in config:
            self.height_ths = config['height_ths']
            
        if 'decoder' in config:
            self.decoder = config['decoder']
            
        if 'beamWidth' in config:
            self.beamWidth = config['beamWidth']
            
        if 'batch_size' in config:
            self.batch_size = config['batch_size']
            
        if 'allowlist' in config:
            self.allowlist = config['allowlist']
            
        if 'blocklist' in config:
            self.blocklist = config['blocklist']
            
        # Update active languages if specified
        if 'languages' in config:
            new_languages = config['languages']
            if new_languages != self.active_languages:
                self.active_languages = new_languages
                # Reinitialize reader with new languages
                if self.is_initialized:
                    self._reinitialize_reader()
                    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available for EasyOCR."""
        try:
            if EASYOCR_AVAILABLE:
                return torch.cuda.is_available()
            else:
                return False
        except (ImportError, AttributeError, NameError):
            return False
    
    def _test_basic_functionality(self) -> None:
        """Test basic EasyOCR functionality with a simple image."""
        try:
            # Create a simple test image with text
            test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Test OCR
            reader_key = tuple(sorted(self.active_languages))
            reader = self.readers[reader_key]
            results = reader.readtext(test_image, paragraph=False)
            
            if not results or not any("TEST" in result[1].upper() for result in results):
                logger.warning("EasyOCR basic functionality test did not return expected result")
            else:
                logger.debug("EasyOCR basic functionality test passed")
                
        except Exception as e:
            logger.warning(f"EasyOCR basic functionality test failed: {str(e)}")
    
    def _initialize_handwriting_support(self) -> None:
        """Initialize handwriting recognition capabilities."""
        try:
            # Check if any active languages support handwriting
            handwriting_langs = [lang for lang in self.active_languages 
                               if lang in self.handwriting_languages]
            
            if handwriting_langs:
                logger.info(f"Handwriting support available for languages: {handwriting_langs}")
                
                # Create specialized reader for handwriting if needed
                if len(handwriting_langs) != len(self.active_languages):
                    hw_reader_key = tuple(sorted(handwriting_langs))
                    if hw_reader_key not in self.readers:
                        self.readers[hw_reader_key] = easyocr.Reader(
                            handwriting_langs,
                            gpu=self._check_gpu_availability(),
                            verbose=False
                        )
                        logger.info(f"Created specialized handwriting reader for: {handwriting_langs}")
            else:
                logger.info("No handwriting support for current language configuration")
                self.handwriting_enabled = False
                
        except Exception as e:
            logger.warning(f"Failed to initialize handwriting support: {str(e)}")
            self.handwriting_enabled = False
            
    def _get_or_create_reader(self, languages: List[str]) -> EasyOCRReader:
        """Get or create a reader for the specified languages."""
        reader_key = tuple(sorted(languages))
        
        with self.reader_lock:
            if reader_key not in self.readers:
                try:
                    self.readers[reader_key] = easyocr.Reader(
                        languages,
                        gpu=self._check_gpu_availability(),
                        verbose=False
                    )
                    logger.info(f"Created new EasyOCR reader for languages: {languages}")
                except Exception as e:
                    logger.error(f"Failed to create reader for {languages}: {str(e)}")
                    raise OCREngineError(f"Failed to create EasyOCR reader: {str(e)}")
            
            return self.readers[reader_key]
    
    def _reinitialize_reader(self) -> None:
        """Reinitialize EasyOCR readers with updated configuration."""
        try:
            # Clear existing readers
            with self.reader_lock:
                self.readers.clear()
            
            # Recreate primary reader
            self._get_or_create_reader(self.active_languages)
            
            # Reinitialize handwriting support if enabled
            if self.handwriting_enabled:
                self._initialize_handwriting_support()
                
            logger.info(f"EasyOCR reinitialized with languages: {self.active_languages}")
        except Exception as e:
            logger.error(f"Failed to reinitialize EasyOCR: {str(e)}")
            raise OCREngineError(f"Failed to reinitialize EasyOCR: {str(e)}")
            
    def _extract_text_impl(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Extract text using EasyOCR with handwriting support and optimizations.
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional parameters
            
        Returns:
            OCR result with extracted text and metadata
        """
        start_time = time.time()
        
        try:
            # Determine if handwriting detection should be used
            detect_handwriting = kwargs.get('detect_handwriting', self.handwriting_enabled)
            language = kwargs.get('language', 'en')
            
            # Select appropriate reader
            reader = self._select_optimal_reader(image, language, detect_handwriting)
            
            # Prepare parameters for EasyOCR
            params = self._prepare_extraction_params(kwargs, detect_handwriting)
            
            # Run EasyOCR with potential handwriting detection
            results = self._run_ocr_with_fallback(reader, image, params, detect_handwriting)
            
            # Process results with enhanced confidence scoring
            ocr_result = self._process_ocr_results(results, detect_handwriting)
            
            # Update processing statistics
            processing_time = time.time() - start_time
            self._update_processing_stats(processing_time, detect_handwriting)
            
            return ocr_result
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {str(e)}")
            raise OCREngineError(
                f"EasyOCR extraction failed: {str(e)}", 
                error_code="EASYOCR_EXTRACTION_FAILED",
                context={"original_error": str(e)}
            )
    
    def _select_optimal_reader(self, image: np.ndarray, language: str, detect_handwriting: bool) -> EasyOCRReader:
        """Select the optimal reader based on image characteristics and requirements."""
        # Determine languages to use
        if language in self.supported_languages:
            target_languages = [language]
        else:
            logger.warning(f"Language {language} not supported, using English")
            target_languages = ['en']
        
        # If handwriting detection is enabled and language supports it
        if detect_handwriting and any(lang in self.handwriting_languages for lang in target_languages):
            # Check if image might contain handwriting
            if self._might_contain_handwriting(image):
                logger.debug("Handwriting detected, using specialized processing")
                self.processing_stats['handwriting_detected'] += 1
        
        return self._get_or_create_reader(target_languages)
    
    def _prepare_extraction_params(self, kwargs: Dict[str, Any], detect_handwriting: bool) -> Dict[str, Any]:
        """Prepare parameters for EasyOCR extraction."""
        params = {
            'width_ths': kwargs.get('width_ths', self.width_ths),
            'height_ths': kwargs.get('height_ths', self.height_ths),
            'decoder': kwargs.get('decoder', self.decoder),
            'beamWidth': kwargs.get('beamWidth', self.beamWidth),
            'batch_size': kwargs.get('batch_size', self.batch_size),
            'workers': kwargs.get('workers', self.workers),
            'paragraph': kwargs.get('paragraph', False)
        }
        
        # Adjust parameters for handwriting
        if detect_handwriting:
            params['width_ths'] = max(0.5, params['width_ths'] - 0.2)  # More lenient for handwriting
            params['height_ths'] = max(0.5, params['height_ths'] - 0.2)
            params['decoder'] = 'beamsearch'  # Better for handwriting
            params['beamWidth'] = max(params['beamWidth'], 10)
        
        # Add character filtering if specified
        if self.allowlist:
            params['allowlist'] = self.allowlist
        if self.blocklist:
            params['blocklist'] = self.blocklist
            
        return params
    
    def _run_ocr_with_fallback(self, reader: EasyOCRReader, image: np.ndarray, 
                              params: Dict[str, Any], detect_handwriting: bool) -> List[Tuple]:
        """Run OCR with fallback strategies for better accuracy."""
        try:
            # Primary OCR attempt
            results = reader.readtext(image, **params)
            
            # If results are poor and handwriting detection is enabled, try with adjusted parameters
            if detect_handwriting and self._are_results_poor(results):
                logger.debug("Poor results detected, trying handwriting-optimized parameters")
                
                # Adjust parameters for handwriting
                hw_params = params.copy()
                hw_params['width_ths'] = 0.4
                hw_params['height_ths'] = 0.4
                hw_params['decoder'] = 'beamsearch'
                hw_params['beamWidth'] = 15
                
                hw_results = reader.readtext(image, **hw_params)
                
                # Use handwriting results if they're better
                if self._compare_results_quality(hw_results, results) > 0:
                    logger.debug("Using handwriting-optimized results")
                    results = hw_results
            
            return results
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise
    
    def _might_contain_handwriting(self, image: np.ndarray) -> bool:
        """Heuristic to detect if image might contain handwriting."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Count edge pixels
            edge_ratio = np.sum(edges > 0) / edges.size
            
            # Handwritten text typically has more irregular edges
            # This is a simple heuristic - could be improved with ML
            return bool(edge_ratio > 0.05)
            
        except Exception as e:
            logger.debug(f"Handwriting detection failed: {str(e)}")
            return False
    
    def _are_results_poor(self, results: List[Tuple]) -> bool:
        """Check if OCR results are of poor quality."""
        if not results:
            return True
        
        # Check average confidence
        confidences = [result[2] for result in results]
        avg_confidence = np.mean(confidences)
        
        return avg_confidence < self.handwriting_threshold
    
    def _compare_results_quality(self, results1: List[Tuple], results2: List[Tuple]) -> int:
        """Compare quality of two result sets. Returns 1 if results1 is better, -1 if results2 is better, 0 if equal."""
        if not results1 and not results2:
            return 0
        if not results1:
            return -1
        if not results2:
            return 1
        
        # Compare average confidence
        conf1 = np.mean([r[2] for r in results1])
        conf2 = np.mean([r[2] for r in results2])
        
        # Compare text length (more text might be better)
        text1_len = sum(len(r[1]) for r in results1)
        text2_len = sum(len(r[1]) for r in results2)
        
        # Weighted comparison
        conf_diff = conf1 - conf2
        len_diff = (text1_len - text2_len) / max(text1_len, text2_len, 1) * 0.1
        
        total_diff = conf_diff + len_diff
        
        if total_diff > 0.05:
            return 1
        elif total_diff < -0.05:
            return -1
        else:
            return 0
    
    def _process_ocr_results(self, results: List[Tuple], detect_handwriting: bool) -> OCRResult:
        """Process OCR results with enhanced confidence scoring."""
        full_text = []
        word_data = []
        bounding_boxes = []
        confidences = []
        
        for bbox, text, confidence in results:
            if not text.strip():
                continue
            
            # Adjust confidence for handwriting
            adjusted_confidence = self._adjust_confidence_for_handwriting(
                confidence, text, detect_handwriting
            )
            
            full_text.append(text)
            confidences.append(adjusted_confidence)
            
            # Convert bbox format: EasyOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            width = int(max(x_coords) - min(x_coords))
            height = int(max(y_coords) - min(y_coords))
            
            bounding_box = BoundingBox(
                x=x, y=y, width=width, height=height, confidence=adjusted_confidence
            )
            
            bounding_boxes.append(bounding_box)
            
            # Create word data with improved word segmentation
            words = self._segment_text_into_words(text, bbox)
            
            for word_info in words:
                word_data.append(WordData(
                    text=word_info['text'],
                    confidence=adjusted_confidence,
                    bounding_box=word_info['bbox']
                ))
        
        # Combine all text
        combined_text = ' '.join(full_text)
        
        # Calculate overall confidence with handwriting adjustment
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        return OCRResult(
            text=combined_text,
            confidence=overall_confidence,
            bounding_boxes=bounding_boxes,
            word_level_data=word_data
        )
    
    def _adjust_confidence_for_handwriting(self, confidence: float, text: str, detect_handwriting: bool) -> float:
        """Adjust confidence score for handwriting recognition."""
        if not detect_handwriting:
            return confidence
        
        # Handwriting recognition is inherently less confident
        # Apply a slight penalty but boost if text looks reasonable
        adjusted = confidence * 0.9
        
        # Boost confidence for reasonable-looking text
        if len(text) > 2 and text.isalnum():
            adjusted = min(1.0, adjusted * 1.1)
        
        return adjusted
    
    def _segment_text_into_words(self, text: str, bbox: List[List[float]]) -> List[Dict[str, Any]]:
        """Segment text into individual words with estimated bounding boxes."""
        words = text.split()
        if not words:
            return []
        
        # Calculate bounding box dimensions
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x = int(min(x_coords))
        y = int(min(y_coords))
        width = int(max(x_coords) - min(x_coords))
        height = int(max(y_coords) - min(y_coords))
        
        # Estimate word positions
        word_infos = []
        total_chars = sum(len(word) for word in words) + len(words) - 1  # Include spaces
        
        current_x = x
        for i, word in enumerate(words):
            # Estimate word width based on character count
            word_width = int((len(word) / total_chars) * width) if total_chars > 0 else width // len(words)
            
            word_bbox = BoundingBox(
                x=current_x,
                y=y,
                width=word_width,
                height=height,
                confidence=0.0  # Will be set by caller
            )
            
            word_infos.append({
                'text': word,
                'bbox': word_bbox
            })
            
            # Move to next word position (add space)
            current_x += word_width + int(width * 0.02)  # Small gap between words
        
        return word_infos
    
    def _update_processing_stats(self, processing_time: float, detect_handwriting: bool) -> None:
        """Update processing statistics."""
        self.processing_stats['total_processed'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['avg_processing_time']
        total_processed = self.processing_stats['total_processed']
        
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (total_processed - 1) + processing_time) / total_processed
        )
            
    def detect_tables(self, image: np.ndarray, **kwargs) -> List[TableRegion]:
        """
        Detect table regions using EasyOCR text detection.
        This is a basic implementation that groups nearby text regions.
        """
        try:
            # Get text detection results with bounding boxes
            results = self.reader.readtext(image, paragraph=False)
            
            if not results:
                return []
                
            # Group text regions that might form tables
            table_regions = self._group_text_into_tables(results)
            
            return table_regions
            
        except Exception as e:
            logger.error(f"EasyOCR table detection failed: {str(e)}")
            return []
            
    def _group_text_into_tables(self, results: List) -> List[TableRegion]:
        """
        Group text detection results into potential table regions.
        """
        if len(results) < 4:  # Need at least 4 text regions for a table
            return []
            
        table_regions = []
        
        # Convert results to a more manageable format
        text_blocks = []
        for bbox, text, confidence in results:
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            text_blocks.append({
                'text': text,
                'x': min(x_coords),
                'y': min(y_coords),
                'width': max(x_coords) - min(x_coords),
                'height': max(y_coords) - min(y_coords),
                'confidence': confidence
            })
            
        # Sort by y-coordinate to identify rows
        text_blocks.sort(key=lambda b: b['y'])
        
        # Group blocks into potential rows
        rows = []
        current_row = []
        current_y = None
        row_height_threshold = 20  # Pixels
        
        for block in text_blocks:
            if current_y is None or abs(block['y'] - current_y) <= row_height_threshold:
                current_row.append(block)
                current_y = block['y']
            else:
                if len(current_row) >= 2:  # At least 2 columns
                    rows.append(current_row)
                current_row = [block]
                current_y = block['y']
                
        # Add last row
        if len(current_row) >= 2:
            rows.append(current_row)
            
        # If we have multiple rows, consider it a table
        if len(rows) >= 2:
            # Calculate table bounding box
            all_blocks = [block for row in rows for block in row]
            
            min_x = min(block['x'] for block in all_blocks)
            max_x = max(block['x'] + block['width'] for block in all_blocks)
            min_y = min(block['y'] for block in all_blocks)
            max_y = max(block['y'] + block['height'] for block in all_blocks)
            
            avg_confidence = np.mean([block['confidence'] for block in all_blocks])
            
            table_regions.append(TableRegion(
                bounding_box=BoundingBox(
                    x=int(min_x),
                    y=int(min_y),
                    width=int(max_x - min_x),
                    height=int(max_y - min_y),
                    confidence=avg_confidence
                ),
                confidence=avg_confidence,
                page_number=1
            ))
            
        return table_regions
        
    def extract_text_batch(self, images: List[np.ndarray], **kwargs) -> List[OCRResult]:
        """
        Extract text from multiple images using batch processing for improved performance.
        
        Args:
            images: List of input images as numpy arrays
            **kwargs: Additional parameters for OCR processing
            
        Returns:
            List of OCR results corresponding to input images
        """
        if not images:
            return []
        
        start_time = time.time()
        
        try:
            # Split into batches
            batch_size = min(kwargs.get('batch_size', self.max_batch_size), len(images))
            batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]
            
            results = []
            
            if len(batches) == 1:
                # Single batch - process directly
                results = self._process_image_batch(batches[0], **kwargs)
            else:
                # Multiple batches - use threading for parallel processing
                with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
                    future_to_batch = {
                        executor.submit(self._process_image_batch, batch, **kwargs): i 
                        for i, batch in enumerate(batches)
                    }
                    
                    batch_results = [None] * len(batches)
                    
                    for future in as_completed(future_to_batch, timeout=self.batch_timeout):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_results[batch_idx] = future.result()
                        except Exception as e:
                            logger.error(f"Batch {batch_idx} processing failed: {str(e)}")
                            # Create error results for this batch
                            batch_results[batch_idx] = [
                                OCRResult(text="", confidence=0.0, bounding_boxes=[], word_level_data=[])
                                for _ in batches[batch_idx]
                            ]
                    
                    # Flatten results
                    for batch_result in batch_results:
                        if batch_result:
                            results.extend(batch_result)
            
            # Update batch processing statistics
            processing_time = time.time() - start_time
            self.processing_stats['batch_processed'] += len(images)
            
            logger.info(f"Batch processed {len(images)} images in {processing_time:.2f}s "
                       f"({len(images)/processing_time:.1f} images/sec)")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise OCREngineError(f"Batch processing failed: {str(e)}")
    
    def _process_image_batch(self, images: List[np.ndarray], **kwargs) -> List[OCRResult]:
        """Process a batch of images."""
        results = []
        
        for image in images:
            try:
                result = self._extract_text_impl(image, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image in batch: {str(e)}")
                # Add empty result to maintain order
                results.append(OCRResult(
                    text="", confidence=0.0, bounding_boxes=[], word_level_data=[]
                ))
        
        return results
    
    def supports_handwriting(self) -> bool:
        """Check if current configuration supports handwriting recognition."""
        return (self.handwriting_enabled and 
                any(lang in self.active_languages for lang in self.handwriting_languages))
    
    def get_handwriting_languages(self) -> List[str]:
        """Get list of languages that support handwriting recognition."""
        return [lang for lang in self.active_languages if lang in self.handwriting_languages]
        
    def preprocess_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Preprocess image for better EasyOCR accuracy.
        """
        try:
            # EasyOCR works well with color images, but we can still apply some preprocessing
            processed = image.copy()
            
            # Apply noise reduction if specified
            if self.config.get('noise_reduction', False):
                if len(processed.shape) == 3:
                    processed = cv2.bilateralFilter(processed, 9, 75, 75)
                else:
                    processed = cv2.GaussianBlur(processed, (3, 3), 0)
                    
            # Enhance contrast if specified
            if self.config.get('contrast_enhancement', False):
                if len(processed.shape) == 3:
                    # Convert to LAB color space for better contrast enhancement
                    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    
                    # Apply CLAHE to L channel
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    
                    # Merge channels and convert back
                    lab = cv2.merge([l, a, b])
                    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                else:
                    # Grayscale image
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    processed = clahe.apply(processed)
                    
            # Resize if specified
            resize_factor = self.config.get('resize_factor', 1.0)
            if resize_factor != 1.0:
                if len(processed.shape) == 3:
                    height, width, _ = processed.shape
                else:
                    height, width = processed.shape
                    
                new_height = int(height * resize_factor)
                new_width = int(width * resize_factor)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                
            return processed
            
        except Exception as e:
            logger.error(f"EasyOCR image preprocessing failed: {str(e)}")
            return image  # Return original image if preprocessing fails
            
    def optimize_for_batch_processing(self, enable: bool = True) -> None:
        """Optimize engine settings for batch processing."""
        if enable:
            self.batch_size = min(8, self.max_batch_size)
            self.workers = min(4, max(1, self.workers))
            logger.info("Enabled batch processing optimizations")
        else:
            self.batch_size = 1
            self.workers = 0
            logger.info("Disabled batch processing optimizations")
    
    def set_handwriting_mode(self, enabled: bool, threshold: float = 0.6) -> None:
        """Enable or disable handwriting recognition mode."""
        self.handwriting_enabled = enabled
        self.handwriting_threshold = max(0.1, min(1.0, threshold))
        
        if enabled:
            self._initialize_handwriting_support()
            logger.info(f"Handwriting recognition enabled (threshold: {threshold})")
        else:
            logger.info("Handwriting recognition disabled")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
        return self.processing_stats.copy()
    
    def reset_processing_stats(self) -> None:
        """Reset processing statistics."""
        self.processing_stats = {
            'total_processed': 0,
            'batch_processed': 0,
            'avg_processing_time': 0.0,
            'handwriting_detected': 0
        }
        logger.info("Processing statistics reset")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive information about EasyOCR engine."""
        info = super().get_engine_info()
        info.update({
            'active_languages': self.active_languages,
            'all_supported_languages': self.supported_languages,
            'handwriting_languages': self.handwriting_languages,
            'gpu_available': self._check_gpu_availability(),
            'supports_handwriting': self.supports_handwriting(),
            'handwriting_enabled': self.handwriting_enabled,
            'handwriting_threshold': self.handwriting_threshold,
            'batch_processing': {
                'max_batch_size': self.max_batch_size,
                'current_batch_size': self.batch_size,
                'batch_timeout': self.batch_timeout
            },
            'ocr_parameters': {
                'width_ths': self.width_ths,
                'height_ths': self.height_ths,
                'decoder': self.decoder,
                'beamWidth': self.beamWidth,
                'workers': self.workers
            },
            'performance_stats': self.get_processing_stats(),
            'active_readers': len(self.readers)
        })
        return info
    
    def cleanup(self) -> None:
        """Cleanup EasyOCR resources."""
        try:
            with self.reader_lock:
                for reader_key, reader in self.readers.items():
                    try:
                        # EasyOCR readers don't have explicit cleanup, but we can clear references
                        del reader
                    except Exception as e:
                        logger.warning(f"Error cleaning up reader {reader_key}: {str(e)}")
                
                self.readers.clear()
            
            # Clear CUDA cache if GPU was used
            if self._check_gpu_availability():
                try:
                    torch.cuda.empty_cache()
                except Exception as e:
                    logger.debug(f"Could not clear CUDA cache: {str(e)}")
            
            super().cleanup()
            logger.info("EasyOCR engine cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during EasyOCR cleanup: {str(e)}")
            super().cleanup()