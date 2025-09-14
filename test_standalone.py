#!/usr/bin/env python3
"""
Test script to verify the standalone Streamlit app works
"""

import sys
import subprocess
from pathlib import Path

def test_standalone_app():
    """Test that the standalone app can be imported and run"""
    
    try:
        # Test import
        print("🧪 Testing standalone app import...")
        import streamlit_app
        print("✅ Standalone app imports successfully")
        
        # Test that required functions exist
        required_functions = [
            'render_header',
            'render_login', 
            'render_sidebar',
            'render_upload_page',
            'render_validation_page',
            'render_chat_page',
            'render_dashboard_page',
            'main'
        ]
        
        for func_name in required_functions:
            if hasattr(streamlit_app, func_name):
                print(f"✅ Function {func_name} exists")
            else:
                print(f"❌ Function {func_name} missing")
                return False
        
        print("✅ All required functions found")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_syntax_check():
    """Run syntax check on the standalone app"""
    
    try:
        print("🧪 Running syntax check...")
        result = subprocess.run([
            sys.executable, "-m", "py_compile", "streamlit_app.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Syntax check passed")
            return True
        else:
            print(f"❌ Syntax errors: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Syntax check failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🔍 Testing OCR Table Analytics Standalone App")
    print("=" * 50)
    
    # Check if file exists
    app_file = Path("streamlit_app.py")
    if not app_file.exists():
        print("❌ streamlit_app.py not found")
        return False
    
    # Run tests
    syntax_ok = run_syntax_check()
    import_ok = test_standalone_app()
    
    if syntax_ok and import_ok:
        print("\n🎉 All tests passed! The standalone app is ready to run.")
        print("\n🚀 To start the app, run:")
        print("   streamlit run streamlit_app.py")
        return True
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)