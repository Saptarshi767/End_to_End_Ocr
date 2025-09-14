#!/usr/bin/env python3
"""
OCR Dependencies Installation Script
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🚀 Installing OCR dependencies...")
    
    # Core packages
    core_packages = [
        "streamlit>=1.28.0",
        "pandas>=1.5.0", 
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.5.0"
    ]
    
    # OCR packages
    ocr_packages = [
        "easyocr>=1.7.0",
        "pytesseract>=0.3.10"
    ]
    
    # Optional packages
    optional_packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "scikit-image>=0.21.0"
    ]
    
    print("\n📦 Installing core packages...")
    for package in core_packages:
        install_package(package)
    
    print("\n🔍 Installing OCR packages...")
    for package in ocr_packages:
        install_package(package)
    
    print("\n⚡ Installing optional packages (for better performance)...")
    for package in optional_packages:
        install_package(package)
    
    print("\n✅ Installation complete!")
    print("\n📋 Next steps:")
    print("1. Make sure Tesseract is installed on your system:")
    print("   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
    print("   - macOS: brew install tesseract")
    print("   - Linux: sudo apt-get install tesseract-ocr")
    print("2. Run: python run_ui.py")

if __name__ == "__main__":
    main()