#!/usr/bin/env python3
"""
Launcher script for Streamlit Live Transcription
Performs basic checks and launches the Streamlit app.
"""

import sys
import subprocess
import importlib.util
import os

def check_dependency(package_name, install_name=None):
    """Check if a package is installed."""
    if install_name is None:
        install_name = package_name
    
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        print(f"❌ {package_name} is not installed")
        print(f"   Install with: pip install {install_name}")
        return False
    else:
        print(f"✅ {package_name} is available")
        return True

def main():
    """Main launcher function."""
    print("🚀 Streamlit Live Transcription Launcher")
    print("=" * 45)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check critical dependencies
    print("\n🔍 Checking dependencies...")
    dependencies = [
        ("streamlit", "streamlit"),
        ("whisper", "openai-whisper"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pyaudio", "pyaudio"),
    ]
    
    all_deps_ok = True
    for package, install_name in dependencies:
        if not check_dependency(package, install_name):
            all_deps_ok = False
    
    if not all_deps_ok:
        print("\n❌ Some dependencies are missing!")
        print("Install them with:")
        print("   pip install -r requirements_streamlit.txt")
        return False
    
    # Check if streamlit_transcription.py exists
    if not os.path.exists("streamlit_transcription.py"):
        print("\n❌ streamlit_transcription.py not found in current directory")
        return False
    else:
        print("✅ streamlit_transcription.py found")
    
    print("\n✅ All checks passed!")
    print("🎤 Starting Streamlit Live Transcription...")
    print("   The app will open in your default browser")
    print("   Press Ctrl+C to stop the server")
    print("-" * 45)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_transcription.py",
            "--server.headless", "false",
            "--server.runOnSave", "true"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 