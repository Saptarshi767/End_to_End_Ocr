#!/usr/bin/env python3
"""
Simple UI Launcher for OCR Table Analytics

This script properly sets up the Python path and launches the Streamlit app.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables
os.environ.setdefault('PYTHONPATH', str(project_root))

if __name__ == "__main__":
    import subprocess
    
    # Get the app file path
    app_file = project_root / "src" / "ui" / "app.py"
    
    if not app_file.exists():
        print(f"❌ App file not found: {app_file}")
        sys.exit(1)
    
    # Launch Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_file),
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("🚀 Launching OCR Table Analytics UI...")
    print(f"📍 URL: http://localhost:8501")
    print("=" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down UI")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)