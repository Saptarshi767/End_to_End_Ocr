#!/usr/bin/env python3
"""
Startup script for OCR Table Analytics UI

This script launches the Streamlit application with proper configuration.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def check_dependencies():
    """Check if required dependencies are installed"""
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('plotly', 'plotly'),
        ('requests', 'requests'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python')
    ]
    
    ocr_packages = [
        ('easyocr', 'easyocr'),
        ('pytesseract', 'pytesseract')
    ]
    
    missing_packages = []
    missing_ocr = []
    
    # Check required packages
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    # Check OCR packages (optional but recommended)
    for import_name, package_name in ocr_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_ocr.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install -r requirements_ocr.txt")
        print("\n🔧 Quick install command:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    if missing_ocr:
        print(f"⚠️  Missing OCR packages: {', '.join(missing_ocr)}")
        print("📦 For full OCR functionality, install: pip install -r requirements_ocr.txt")
        print("🔧 Or run: python install_ocr_deps.py")
        print("📝 The app will work with limited functionality without OCR packages")
    else:
        print("✅ All OCR packages are available")
    
    print("✅ All required packages are installed")
    return True


def setup_environment():
    """Setup environment variables and configuration"""
    
    # Load environment variables from .env file if it exists
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print(f"📄 Loaded environment from {env_file}")
    
    # Set default environment variables if not already set
    defaults = {
        'API_BASE_URL': 'http://localhost:8000',
        'UI_HOST': '0.0.0.0',
        'UI_PORT': '8501',
        'UI_DEBUG': 'false'
    }
    
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
    
    print(f"🌐 API Base URL: {os.environ.get('API_BASE_URL')}")
    print(f"🖥️  UI Host: {os.environ.get('UI_HOST')}:{os.environ.get('UI_PORT')}")


def create_streamlit_config():
    """Create Streamlit configuration file"""
    
    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "config.toml"
    
    config_content = """
[global]
developmentMode = false
showWarningOnDirectExecution = false

[server]
headless = true
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
primaryColor = "#0066CC"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[logger]
level = "info"
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content.strip())
    
    print(f"⚙️  Created Streamlit config at {config_file}")


def run_streamlit_app(host='0.0.0.0', port=8501, debug=False):
    """Run the Streamlit application"""
    
    # Try standalone app first (more reliable), then fallback to full app
    standalone_file = Path(__file__).parent / "streamlit_app.py"
    app_file = Path(__file__).parent / "src" / "ui" / "app.py"
    
    if standalone_file.exists():
        selected_app = standalone_file
        print("🎯 Using standalone application (recommended)")
    elif app_file.exists():
        selected_app = app_file
        print("🎯 Using full application with all features")
    else:
        print(f"❌ No application file found")
        return False
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(selected_app),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    if debug:
        cmd.extend(["--logger.level", "debug"])
    
    print(f"🚀 Starting OCR Table Analytics UI...")
    print(f"📍 URL: http://{host}:{port}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # Run streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down OCR Table Analytics UI")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start application: {e}")
        return False
    
    return True


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="OCR Table Analytics UI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ui.py                    # Run with default settings
  python run_ui.py --port 8502        # Run on custom port
  python run_ui.py --debug            # Run in debug mode
  python run_ui.py --host 127.0.0.1   # Run on localhost only
        """
    )
    
    parser.add_argument(
        '--host',
        default=os.environ.get('UI_HOST', '0.0.0.0'),
        help='Host to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=int(os.environ.get('UI_PORT', 8501)),
        help='Port to bind to (default: 8501)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        default=os.environ.get('UI_DEBUG', 'false').lower() == 'true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency and environment checks'
    )
    
    args = parser.parse_args()
    
    print("🔍 OCR Table Analytics UI Launcher")
    print("=" * 40)
    
    # Check dependencies
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        
        # Setup environment
        setup_environment()
        
        # Create Streamlit config
        create_streamlit_config()
    
    # Run the application
    success = run_streamlit_app(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()