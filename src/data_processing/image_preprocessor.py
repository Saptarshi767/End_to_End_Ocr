"""
Image preprocessing module for enhancing document images before OCR processing.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

from src.core.exceptions import DocumentProcessingError


class PreprocessingMethod(Enum):
    """Available preprocessing methods."""
    NOISE_REDUCTION = "noise_reduction"
    CONTRAST_ENHANCEMENT = "contrast_enhancement"
    BRIGHTNESS_ADJUSTMENT = "brightness_adjustment"
    SHARPENING = "sharpening"
    DESKEWING = "deskewing"
    BINARIZATION = "binarization"
    MORPHOLOGICAL_OPERATIONS = "morphological_operations"


@dataclass
class ImageQualityMetrics:
    """Image quality assessment metrics."""
    brightness: float
    contrast: float
    sharpness: float
    noise_level: float
    skew_angle: float
    resolution_dpi: Optional[int] = None
    overall_score: float = 0.0
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing."""
    target_dpi: int = 300
    enable_noise_reduction: bool = True
    enable_contrast_enhancement: bool = True
    enable_brightness_adjustment: bool = True
    enable_sharpening: bool = True
    enable_deskewing: bool = True
    enable_binarization: bool = False  # Only for very poor quality images
    contrast_factor: float = 1.2
    brightness_factor: float = 1.0
    sharpness_factor: float = 1.1
    noise_reduction_strength: int = 3
    binarization_threshold: int = 128
    max_skew_angle: float = 45.0


class ImagePreprocessor:
    """
    Handles image preprocessing for optimal OCR performance.
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize image preprocessor.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)

    def preprocess_image(self, image: np.ndarray, quality_metrics: Optional[ImageQualityMetrics] = None) -> np.ndarray:
        """
        Apply comprehensive image preprocessing for better OCR accuracy.
        
        Args:
            image: Input image as numpy array
            quality_metrics: Pre-computed quality metrics (optional)
            
        Returns:
            Preprocessed image as numpy array
            
        Raises:
            DocumentProcessingError: If preprocessing fails
        """
        try:
            if quality_metrics is None:
                quality_metrics = self.assess_image_quality(image)

            processed_image = image.copy()

            # Apply preprocessing steps based on quality assessment
            if self.config.enable_noise_reduction and quality_metrics.noise_level > 0.3:
                processed_image = self._reduce_noise(processed_image)

            if self.config.enable_deskewing and abs(quality_metrics.skew_angle) > 1.0:
                processed_image = self._deskew_image(processed_image, quality_metrics.skew_angle)

            if self.config.enable_contrast_enhancement and quality_metrics.contrast < 0.5:
                processed_image = self._enhance_contrast(processed_image)

            if self.config.enable_brightness_adjustment and (quality_metrics.brightness < 0.3 or quality_metrics.brightness > 0.8):
                processed_image = self._adjust_brightness(processed_image, quality_metrics.brightness)

            if self.config.enable_sharpening and quality_metrics.sharpness < 0.5:
                processed_image = self._sharpen_image(processed_image)

            if self.config.enable_binarization and quality_metrics.overall_score < 0.3:
                processed_image = self._binarize_image(processed_image)

            # Apply morphological operations for cleanup
            processed_image = self._apply_morphological_operations(processed_image)

            return processed_image

        except Exception as e:
            raise DocumentProcessingError(f"Image preprocessing failed: {str(e)}")

    def assess_image_quality(self, image: np.ndarray) -> ImageQualityMetrics:
        """
        Assess image quality and provide recommendations for preprocessing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            ImageQualityMetrics object with quality assessment
        """
        try:
            # Convert to PIL Image for some operations
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)

            # Calculate quality metrics
            brightness = self._calculate_brightness(image)
            contrast = self._calculate_contrast(image)
            sharpness = self._calculate_sharpness(image)
            noise_level = self._calculate_noise_level(image)
            skew_angle = self._detect_skew_angle(image)

            # Calculate overall quality score
            overall_score = self._calculate_overall_score(brightness, contrast, sharpness, noise_level, abs(skew_angle))

            # Generate recommendations
            recommendations = self._generate_recommendations(brightness, contrast, sharpness, noise_level, skew_angle)

            return ImageQualityMetrics(
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                noise_level=noise_level,
                skew_angle=skew_angle,
                overall_score=overall_score,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {str(e)}")
            return ImageQualityMetrics(
                brightness=0.5,
                contrast=0.5,
                sharpness=0.5,
                noise_level=0.5,
                skew_angle=0.0,
                overall_score=0.5,
                recommendations=["Quality assessment failed - applying default preprocessing"]
            )

    def convert_format(self, image: np.ndarray, target_format: str = 'RGB') -> np.ndarray:
        """
        Convert image format for processing.
        
        Args:
            image: Input image
            target_format: Target format ('RGB', 'GRAY', 'BGR')
            
        Returns:
            Converted image
        """
        try:
            if target_format == 'GRAY' and len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif target_format == 'RGB' and len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif target_format == 'BGR' and len(image.shape) == 3:
                return image  # Assuming input is already BGR
            else:
                return image
        except Exception as e:
            raise DocumentProcessingError(f"Format conversion failed: {str(e)}")

    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using bilateral filter."""
        if len(image.shape) == 3:
            return cv2.bilateralFilter(image, self.config.noise_reduction_strength * 3, 80, 80)
        else:
            return cv2.bilateralFilter(image, self.config.noise_reduction_strength * 3, 80, 80)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE."""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def _adjust_brightness(self, image: np.ndarray, current_brightness: float) -> np.ndarray:
        """Adjust image brightness to optimal level."""
        target_brightness = 0.6  # Target brightness level
        adjustment_factor = target_brightness / max(current_brightness, 0.1)
        
        # Limit adjustment to prevent over-correction
        adjustment_factor = max(0.5, min(2.0, adjustment_factor))
        
        return cv2.convertScaleAbs(image, alpha=adjustment_factor, beta=0)

    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter to improve text clarity."""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def _deskew_image(self, image: np.ndarray, skew_angle: float) -> np.ndarray:
        """Correct image skew by rotating."""
        if abs(skew_angle) < 0.5:  # Skip if skew is minimal
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        
        # Calculate new dimensions
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # Adjust rotation matrix for new dimensions
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        return cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for binarization."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

    def _apply_morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological operations for cleanup."""
        if len(image.shape) == 3:
            return image  # Skip for color images
            
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness of image."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.mean(gray) / 255.0

    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate contrast using standard deviation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return np.std(gray) / 255.0

    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate sharpness using Laplacian variance."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return min(laplacian_var / 1000.0, 1.0)  # Normalize to 0-1 range

    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Estimate noise level using high-frequency content."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur and calculate difference
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        return np.mean(noise) / 255.0

    def _detect_skew_angle(self, image: np.ndarray) -> float:
        """Detect skew angle using Hough line transform."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = np.degrees(theta) - 90
                    if abs(angle) < self.config.max_skew_angle:
                        angles.append(angle)
                
                if angles:
                    return np.median(angles)
            
            return 0.0
        except Exception:
            return 0.0

    def _calculate_overall_score(self, brightness: float, contrast: float, 
                               sharpness: float, noise_level: float, skew_angle: float) -> float:
        """Calculate overall image quality score."""
        # Ideal ranges
        brightness_score = 1.0 - abs(brightness - 0.6) / 0.6
        contrast_score = min(contrast * 2, 1.0)
        sharpness_score = min(sharpness * 2, 1.0)
        noise_score = 1.0 - noise_level
        skew_score = 1.0 - min(abs(skew_angle) / 10.0, 1.0)
        
        # Weighted average
        weights = [0.2, 0.25, 0.25, 0.2, 0.1]
        scores = [brightness_score, contrast_score, sharpness_score, noise_score, skew_score]
        
        return sum(w * s for w, s in zip(weights, scores))

    def _generate_recommendations(self, brightness: float, contrast: float, 
                                sharpness: float, noise_level: float, skew_angle: float) -> List[str]:
        """Generate preprocessing recommendations based on quality metrics."""
        recommendations = []
        
        if brightness < 0.3:
            recommendations.append("Image is too dark - brightness adjustment recommended")
        elif brightness > 0.8:
            recommendations.append("Image is too bright - brightness adjustment recommended")
            
        if contrast < 0.3:
            recommendations.append("Low contrast detected - contrast enhancement recommended")
            
        if sharpness < 0.3:
            recommendations.append("Image appears blurry - sharpening recommended")
            
        if noise_level > 0.4:
            recommendations.append("High noise level detected - noise reduction recommended")
            
        if abs(skew_angle) > 2.0:
            recommendations.append(f"Image is skewed by {skew_angle:.1f}° - deskewing recommended")
            
        if not recommendations:
            recommendations.append("Image quality is good - minimal preprocessing needed")
            
        return recommendations