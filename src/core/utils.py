"""
Utility functions for the OCR Table Analytics system.
"""

import os
import hashlib
import mimetypes
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("ocr_analytics")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash of file contents."""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get comprehensive file information."""
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = path.stat()
    mime_type, _ = mimetypes.guess_type(file_path)
    
    return {
        'filename': path.name,
        'file_path': str(path.absolute()),
        'file_size': stat.st_size,
        'mime_type': mime_type or 'application/octet-stream',
        'created_at': datetime.fromtimestamp(stat.st_ctime),
        'modified_at': datetime.fromtimestamp(stat.st_mtime),
        'file_hash': get_file_hash(file_path)
    }


def is_supported_format(file_path: str, supported_formats: List[str]) -> bool:
    """Check if file format is supported."""
    file_extension = Path(file_path).suffix.lower().lstrip('.')
    return file_extension in [fmt.lower() for fmt in supported_formats]


def ensure_directory(directory_path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)


def clean_filename(filename: str) -> str:
    """Clean filename by removing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def validate_api_key(api_key: str, provider: str = "openai") -> bool:
    """Validate API key format."""
    if not api_key:
        return False
    
    if provider == "openai":
        return api_key.startswith("sk-") and len(api_key) > 20
    elif provider == "claude":
        return len(api_key) > 20  # Basic length check
    
    return len(api_key) > 10  # Generic validation


def sanitize_column_name(column_name: str) -> str:
    """Sanitize column name for database/query usage."""
    # Remove special characters and replace with underscores
    sanitized = ''.join(c if c.isalnum() else '_' for c in column_name)
    
    # Remove consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'col_' + sanitized
    
    # Handle empty names
    if not sanitized:
        sanitized = 'unnamed_column'
    
    return sanitized.lower()


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dictionaries(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries recursively."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dictionaries(result[key], value)
        else:
            result[key] = value
    
    return result


def calculate_confidence_score(scores: List[float]) -> float:
    """Calculate overall confidence score from individual scores."""
    if not scores:
        return 0.0
    
    # Use weighted average with higher weight for higher scores
    weights = [score for score in scores]
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    weight_sum = sum(weights)
    
    return weighted_sum / weight_sum if weight_sum > 0 else 0.0


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract numeric values from text."""
    import re
    
    # Pattern to match numbers (including decimals and negatives)
    number_pattern = r'-?\d+\.?\d*'
    matches = re.findall(number_pattern, text)
    
    numbers = []
    for match in matches:
        try:
            if '.' in match:
                numbers.append(float(match))
            else:
                numbers.append(float(match))
        except ValueError:
            continue
    
    return numbers


def detect_date_format(date_string: str) -> Optional[str]:
    """Detect date format from string."""
    import re
    
    date_patterns = {
        r'\d{4}-\d{2}-\d{2}': '%Y-%m-%d',
        r'\d{2}/\d{2}/\d{4}': '%m/%d/%Y',
        r'\d{2}-\d{2}-\d{4}': '%m-%d-%Y',
        r'\d{2}\.\d{2}\.\d{4}': '%m.%d.%Y',
        r'\d{4}/\d{2}/\d{2}': '%Y/%m/%d',
    }
    
    for pattern, format_str in date_patterns.items():
        if re.match(pattern, date_string.strip()):
            return format_str
    
    return None


def create_temp_file(suffix: str = "", prefix: str = "ocr_", directory: str = None) -> str:
    """Create temporary file and return path."""
    import tempfile
    
    if directory:
        ensure_directory(directory)
    
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=directory)
    os.close(fd)  # Close file descriptor, keep the file
    
    return path


def cleanup_temp_files(file_paths: List[str]) -> None:
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.warning(f"Could not remove temp file {file_path}: {e}")


def get_system_info() -> Dict[str, Any]:
    """Get system information for debugging."""
    import platform
    import psutil
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'disk_usage': psutil.disk_usage('/').percent
    }