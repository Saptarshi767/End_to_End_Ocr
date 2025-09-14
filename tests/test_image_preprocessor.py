"""
Unit tests for image preprocessor module.
"""

import numpy as np
import cv2
import pytest
from unittest.mock import Mock, patch
from PIL import Image

from src.data_processing.image_preprocessor import (
    ImagePreprocessor, ImageQualityMetrics, PreprocessingConfig, PreprocessingMethod
)
from src.core.exceptions import DocumentProcessingError


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = ImagePreprocessor()
        
        # Create test images
        self.test_image_color = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(0, 255, (400, 600), dtype=np.uint8)
        
        # Create a low-quality test image
        self.low_quality_image = np.full((400, 600), 50, dtype=np.uint8)  # Very dark
        
        # Create a noisy test image
        noise = np.random.randint(0, 50, (400, 600), dtype=np.uint8)
        self.noisy_image = np.clip(self.test_image_gray.astype(int) + noise, 0, 255).astype(np.uint8)

    def test_preprocess_image_default_config(self):
        """Test image preprocessing with default configuration."""
        result = self.preprocessor.preprocess_image(self.test_image_color)
        
        assert result is not None
        assert result.shape == self.test_image_color.shape
        assert result.dtype == np.uint8

    def test_preprocess_image_with_quality_metrics(self):
        """Test image preprocessing with pre-computed quality metrics."""
        quality_metrics = ImageQualityMetrics(
            brightness=0.2,  # Too dark
            contrast=0.3,    # Low contrast
            sharpness=0.2,   # Blurry
            noise_level=0.5, # Noisy
            skew_angle=5.0   # Skewed
        )
        
        result = self.preprocessor.preprocess_image(self.test_image_color, quality_metrics)
        
        assert result is not None
        assert result.shape == self.test_image_color.shape

    def test_preprocess_image_grayscale(self):
        """Test preprocessing grayscale image."""
        result = self.preprocessor.preprocess_image(self.test_image_gray)
        
        assert result is not None
        assert len(result.shape) == 2  # Should remain grayscale

    def test_preprocess_image_error_handling(self):
        """Test error handling in image preprocessing."""
        # Test with invalid image data
        invalid_image = np.array([])
        
        with pytest.raises(DocumentProcessingError):
            self.preprocessor.preprocess_image(invalid_image)

    def test_assess_image_quality_color_image(self):
        """Test quality assessment for color image."""
        metrics = self.preprocessor.assess_image_quality(self.test_image_color)
        
        assert isinstance(metrics, ImageQualityMetrics)
        assert 0 <= metrics.brightness <= 1
        assert 0 <= metrics.contrast <= 1
        assert 0 <= metrics.sharpness <= 1
        assert 0 <= metrics.noise_level <= 1
        assert -45 <= metrics.skew_angle <= 45
        assert 0 <= metrics.overall_score <= 1
        assert isinstance(metrics.recommendations, list)

    def test_assess_image_quality_grayscale_image(self):
        """Test quality assessment for grayscale image."""
        metrics = self.preprocessor.assess_image_quality(self.test_image_gray)
        
        assert isinstance(metrics, ImageQualityMetrics)
        assert 0 <= metrics.brightness <= 1
        assert 0 <= metrics.contrast <= 1

    def test_assess_image_quality_low_quality_image(self):
        """Test quality assessment for low quality image."""
        metrics = self.preprocessor.assess_image_quality(self.low_quality_image)
        
        assert metrics.brightness < 0.3  # Should detect low brightness
        assert metrics.overall_score < 0.5  # Should have low overall score
        assert len(metrics.recommendations) > 0

    def test_convert_format_bgr_to_rgb(self):
        """Test format conversion from BGR to RGB."""
        result = self.preprocessor.convert_format(self.test_image_color, 'RGB')
        
        assert result.shape == self.test_image_color.shape
        # Colors should be swapped (BGR -> RGB)
        np.testing.assert_array_equal(result[:, :, 0], self.test_image_color[:, :, 2])

    def test_convert_format_color_to_gray(self):
        """Test format conversion from color to grayscale."""
        result = self.preprocessor.convert_format(self.test_image_color, 'GRAY')
        
        assert len(result.shape) == 2
        assert result.shape[:2] == self.test_image_color.shape[:2]

    def test_convert_format_same_format(self):
        """Test format conversion when target format is same as input."""
        result = self.preprocessor.convert_format(self.test_image_gray, 'GRAY')
        
        np.testing.assert_array_equal(result, self.test_image_gray)

    def test_convert_format_error_handling(self):
        """Test error handling in format conversion."""
        with pytest.raises(DocumentProcessingError):
            self.preprocessor.convert_format(np.array([]), 'RGB')

    def test_reduce_noise_color_image(self):
        """Test noise reduction on color image."""
        result = self.preprocessor._reduce_noise(self.noisy_image)
        
        assert result.shape == self.noisy_image.shape
        # Noise reduction should smooth the image
        assert np.std(result) <= np.std(self.noisy_image)

    def test_enhance_contrast_color_image(self):
        """Test contrast enhancement on color image."""
        result = self.preprocessor._enhance_contrast(self.test_image_color)
        
        assert result.shape == self.test_image_color.shape
        assert result.dtype == np.uint8

    def test_enhance_contrast_grayscale_image(self):
        """Test contrast enhancement on grayscale image."""
        result = self.preprocessor._enhance_contrast(self.test_image_gray)
        
        assert result.shape == self.test_image_gray.shape
        assert result.dtype == np.uint8

    def test_adjust_brightness_dark_image(self):
        """Test brightness adjustment on dark image."""
        dark_image = np.full((100, 100), 30, dtype=np.uint8)
        result = self.preprocessor._adjust_brightness(dark_image, 0.1)
        
        assert np.mean(result) > np.mean(dark_image)

    def test_adjust_brightness_bright_image(self):
        """Test brightness adjustment on bright image."""
        bright_image = np.full((100, 100), 200, dtype=np.uint8)
        result = self.preprocessor._adjust_brightness(bright_image, 0.9)
        
        assert np.mean(result) < np.mean(bright_image)

    def test_sharpen_image(self):
        """Test image sharpening."""
        result = self.preprocessor._sharpen_image(self.test_image_gray)
        
        assert result.shape == self.test_image_gray.shape
        assert result.dtype == np.uint8

    def test_deskew_image_minimal_skew(self):
        """Test deskewing with minimal skew angle."""
        result = self.preprocessor._deskew_image(self.test_image_gray, 0.3)
        
        # Should return original image for minimal skew
        np.testing.assert_array_equal(result, self.test_image_gray)

    def test_deskew_image_significant_skew(self):
        """Test deskewing with significant skew angle."""
        result = self.preprocessor._deskew_image(self.test_image_gray, 5.0)
        
        assert result is not None
        # Dimensions might change due to rotation
        assert result.dtype == np.uint8

    def test_binarize_image_color(self):
        """Test binarization of color image."""
        result = self.preprocessor._binarize_image(self.test_image_color)
        
        assert len(result.shape) == 2  # Should be grayscale
        assert result.dtype == np.uint8
        # Should contain only 0 and 255 values
        unique_values = np.unique(result)
        assert len(unique_values) <= 2

    def test_binarize_image_grayscale(self):
        """Test binarization of grayscale image."""
        result = self.preprocessor._binarize_image(self.test_image_gray)
        
        assert result.shape == self.test_image_gray.shape
        assert result.dtype == np.uint8

    def test_apply_morphological_operations_color(self):
        """Test morphological operations on color image."""
        result = self.preprocessor._apply_morphological_operations(self.test_image_color)
        
        # Should return original for color images
        np.testing.assert_array_equal(result, self.test_image_color)

    def test_apply_morphological_operations_grayscale(self):
        """Test morphological operations on grayscale image."""
        result = self.preprocessor._apply_morphological_operations(self.test_image_gray)
        
        assert result.shape == self.test_image_gray.shape
        assert result.dtype == np.uint8

    def test_calculate_brightness(self):
        """Test brightness calculation."""
        # Test with known brightness values
        dark_image = np.zeros((100, 100), dtype=np.uint8)
        bright_image = np.full((100, 100), 255, dtype=np.uint8)
        
        dark_brightness = self.preprocessor._calculate_brightness(dark_image)
        bright_brightness = self.preprocessor._calculate_brightness(bright_image)
        
        assert dark_brightness == 0.0
        assert bright_brightness == 1.0

    def test_calculate_contrast(self):
        """Test contrast calculation."""
        # Low contrast image (all same value)
        low_contrast = np.full((100, 100), 128, dtype=np.uint8)
        # High contrast image (black and white)
        high_contrast = np.zeros((100, 100), dtype=np.uint8)
        high_contrast[:50, :] = 255
        
        low_contrast_value = self.preprocessor._calculate_contrast(low_contrast)
        high_contrast_value = self.preprocessor._calculate_contrast(high_contrast)
        
        assert low_contrast_value < high_contrast_value

    def test_calculate_sharpness(self):
        """Test sharpness calculation."""
        # Create a sharp image with edges
        sharp_image = np.zeros((100, 100), dtype=np.uint8)
        sharp_image[40:60, 40:60] = 255
        
        sharpness = self.preprocessor._calculate_sharpness(sharp_image)
        
        assert sharpness > 0

    def test_calculate_noise_level(self):
        """Test noise level calculation."""
        # Clean image
        clean_image = np.full((100, 100), 128, dtype=np.uint8)
        
        clean_noise = self.preprocessor._calculate_noise_level(clean_image)
        noisy_noise = self.preprocessor._calculate_noise_level(self.noisy_image)
        
        assert noisy_noise > clean_noise

    def test_detect_skew_angle_no_lines(self):
        """Test skew detection with no clear lines."""
        # Random noise image
        noise_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        skew_angle = self.preprocessor._detect_skew_angle(noise_image)
        
        assert isinstance(skew_angle, float)
        assert -45 <= skew_angle <= 45

    def test_detect_skew_angle_error_handling(self):
        """Test skew detection error handling."""
        # Test with invalid image
        with patch('cv2.Canny', side_effect=Exception("Test error")):
            skew_angle = self.preprocessor._detect_skew_angle(self.test_image_gray)
            assert skew_angle == 0.0

    def test_calculate_overall_score(self):
        """Test overall quality score calculation."""
        score = self.preprocessor._calculate_overall_score(
            brightness=0.6,
            contrast=0.5,
            sharpness=0.5,
            noise_level=0.2,
            skew_angle=1.0
        )
        
        assert 0 <= score <= 1

    def test_generate_recommendations_good_quality(self):
        """Test recommendations for good quality image."""
        recommendations = self.preprocessor._generate_recommendations(
            brightness=0.6,
            contrast=0.6,
            sharpness=0.6,
            noise_level=0.2,
            skew_angle=0.5
        )
        
        assert "good" in recommendations[0].lower()

    def test_generate_recommendations_poor_quality(self):
        """Test recommendations for poor quality image."""
        recommendations = self.preprocessor._generate_recommendations(
            brightness=0.1,  # Too dark
            contrast=0.2,    # Low contrast
            sharpness=0.2,   # Blurry
            noise_level=0.6, # Noisy
            skew_angle=5.0   # Skewed
        )
        
        assert len(recommendations) > 1
        assert any("dark" in rec.lower() for rec in recommendations)
        assert any("contrast" in rec.lower() for rec in recommendations)
        assert any("blur" in rec.lower() for rec in recommendations)
        assert any("noise" in rec.lower() for rec in recommendations)
        assert any("skew" in rec.lower() for rec in recommendations)

    def test_preprocessing_config_defaults(self):
        """Test preprocessing configuration defaults."""
        config = PreprocessingConfig()
        
        assert config.target_dpi == 300
        assert config.enable_noise_reduction is True
        assert config.enable_contrast_enhancement is True
        assert config.enable_brightness_adjustment is True
        assert config.enable_sharpening is True
        assert config.enable_deskewing is True
        assert config.enable_binarization is False

    def test_preprocessing_config_custom(self):
        """Test custom preprocessing configuration."""
        config = PreprocessingConfig(
            target_dpi=150,
            enable_noise_reduction=False,
            contrast_factor=1.5
        )
        
        assert config.target_dpi == 150
        assert config.enable_noise_reduction is False
        assert config.contrast_factor == 1.5

    def test_image_quality_metrics_post_init(self):
        """Test ImageQualityMetrics post-initialization."""
        metrics = ImageQualityMetrics(
            brightness=0.5,
            contrast=0.5,
            sharpness=0.5,
            noise_level=0.3,
            skew_angle=2.0
        )
        
        assert metrics.recommendations == []

    def test_preprocessing_method_enum(self):
        """Test PreprocessingMethod enumeration."""
        assert PreprocessingMethod.NOISE_REDUCTION.value == "noise_reduction"
        assert PreprocessingMethod.CONTRAST_ENHANCEMENT.value == "contrast_enhancement"
        assert PreprocessingMethod.BRIGHTNESS_ADJUSTMENT.value == "brightness_adjustment"
        assert PreprocessingMethod.SHARPENING.value == "sharpening"
        assert PreprocessingMethod.DESKEWING.value == "deskewing"
        assert PreprocessingMethod.BINARIZATION.value == "binarization"
        assert PreprocessingMethod.MORPHOLOGICAL_OPERATIONS.value == "morphological_operations"