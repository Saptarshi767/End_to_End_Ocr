"""
Demonstration of enhanced EasyOCR engine features.
This script shows the new capabilities without requiring actual EasyOCR installation.
"""

import numpy as np
import cv2
from src.ocr.easyocr_engine import EasyOCREngine


def demonstrate_enhanced_features():
    """Demonstrate the enhanced EasyOCR engine features."""
    print("=== Enhanced EasyOCR Engine Demonstration ===\n")
    
    # Create engine instance
    engine = EasyOCREngine(confidence_threshold=0.7)
    print(f"Created EasyOCR engine: {engine.name}")
    print(f"Default confidence threshold: {engine.confidence_threshold}")
    print(f"Handwriting support enabled: {engine.handwriting_enabled}")
    print(f"Active languages: {engine.active_languages}")
    print()
    
    # Demonstrate multi-language configuration
    print("=== Multi-Language Support ===")
    print(f"All supported languages ({len(engine.supported_languages)}): {engine.supported_languages[:10]}...")
    print(f"Handwriting languages ({len(engine.handwriting_languages)}): {engine.handwriting_languages}")
    
    # Configure for multiple languages
    engine.configure({
        'languages': ['en', 'ch_sim', 'fr', 'de'],
        'width_ths': 0.8,
        'height_ths': 0.8
    })
    print(f"Configured for languages: {engine.active_languages}")
    print(f"Supports handwriting: {engine.supports_handwriting()}")
    print(f"Handwriting languages available: {engine.get_handwriting_languages()}")
    print()
    
    # Demonstrate handwriting mode configuration
    print("=== Handwriting Recognition Configuration ===")
    engine.set_handwriting_mode(True, threshold=0.6)
    print(f"Handwriting mode enabled: {engine.handwriting_enabled}")
    print(f"Handwriting threshold: {engine.handwriting_threshold}")
    
    # Test handwriting detection heuristic
    test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Sample Text", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    might_be_handwriting = engine._might_contain_handwriting(test_image)
    print(f"Sample image might contain handwriting: {might_be_handwriting}")
    print()
    
    # Demonstrate batch processing optimization
    print("=== Batch Processing Optimization ===")
    print(f"Default batch size: {engine.batch_size}")
    print(f"Max batch size: {engine.max_batch_size}")
    
    engine.optimize_for_batch_processing(True)
    print(f"After optimization - batch size: {engine.batch_size}")
    print(f"Workers: {engine.workers}")
    
    engine.optimize_for_batch_processing(False)
    print(f"After disabling - batch size: {engine.batch_size}")
    print()
    
    # Demonstrate confidence adjustment
    print("=== Confidence Adjustment for Handwriting ===")
    normal_conf = engine._adjust_confidence_for_handwriting(0.9, "normal text", False)
    hw_conf = engine._adjust_confidence_for_handwriting(0.9, "handwritten", True)
    reasonable_conf = engine._adjust_confidence_for_handwriting(0.8, "reasonable", True)
    
    print(f"Normal text confidence (0.9): {normal_conf:.3f}")
    print(f"Handwriting confidence (0.9): {hw_conf:.3f}")
    print(f"Reasonable handwriting (0.8): {reasonable_conf:.3f}")
    print()
    
    # Demonstrate text segmentation
    print("=== Text Segmentation ===")
    text = "Hello world testing"
    bbox = [[10, 10], [150, 10], [150, 30], [10, 30]]
    word_infos = engine._segment_text_into_words(text, bbox)
    
    print(f"Original text: '{text}'")
    print("Word segmentation:")
    for i, word_info in enumerate(word_infos):
        bbox = word_info['bbox']
        print(f"  {i+1}. '{word_info['text']}' at ({bbox.x}, {bbox.y}) size {bbox.width}x{bbox.height}")
    print()
    
    # Demonstrate results quality comparison
    print("=== Results Quality Comparison ===")
    good_results = [
        ([[10, 10], [50, 10], [50, 30], [10, 30]], "Good", 0.9),
        ([[60, 10], [100, 10], [100, 30], [60, 30]], "Text", 0.85)
    ]
    poor_results = [
        ([[10, 10], [30, 10], [30, 30], [10, 30]], "Bad", 0.6)
    ]
    
    comparison = engine._compare_results_quality(good_results, poor_results)
    print(f"Good vs Poor results comparison: {comparison} (1=better, -1=worse, 0=equal)")
    print()
    
    # Demonstrate processing statistics
    print("=== Processing Statistics ===")
    stats = engine.get_processing_stats()
    print("Initial stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Simulate some processing
    engine._update_processing_stats(0.5, False)
    engine._update_processing_stats(0.3, True)
    engine.processing_stats['handwriting_detected'] += 1
    engine.processing_stats['batch_processed'] += 5
    
    updated_stats = engine.get_processing_stats()
    print("\nAfter simulated processing:")
    for key, value in updated_stats.items():
        print(f"  {key}: {value}")
    print()
    
    # Demonstrate comprehensive engine info
    print("=== Comprehensive Engine Information ===")
    info = engine.get_engine_info()
    
    print("Engine Info Summary:")
    print(f"  Name: {info['name']}")
    print(f"  Confidence threshold: {info['confidence_threshold']}")
    print(f"  Active languages: {info['active_languages']}")
    print(f"  GPU available: {info['gpu_available']}")
    print(f"  Handwriting enabled: {info['handwriting_enabled']}")
    print(f"  Active readers: {info['active_readers']}")
    
    print("\nBatch Processing Config:")
    batch_info = info['batch_processing']
    for key, value in batch_info.items():
        print(f"  {key}: {value}")
    
    print("\nOCR Parameters:")
    ocr_params = info['ocr_parameters']
    for key, value in ocr_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Demonstrate cleanup
    print("=== Resource Cleanup ===")
    print(f"Engine initialized: {engine.is_initialized}")
    print(f"Active readers before cleanup: {len(engine.readers)}")
    
    engine.cleanup()
    print(f"Engine initialized after cleanup: {engine.is_initialized}")
    print(f"Active readers after cleanup: {len(engine.readers)}")
    print()
    
    print("=== Demonstration Complete ===")
    print("Enhanced EasyOCR engine features:")
    print("✓ Multi-language support with 40+ languages")
    print("✓ Handwritten text recognition for 9 languages")
    print("✓ Batch processing optimization")
    print("✓ Advanced confidence scoring")
    print("✓ Intelligent text segmentation")
    print("✓ Performance monitoring and statistics")
    print("✓ Comprehensive configuration options")
    print("✓ Robust error handling and fallback mechanisms")


if __name__ == "__main__":
    demonstrate_enhanced_features()