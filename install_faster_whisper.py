#!/usr/bin/env python3
"""
Installation Helper for Faster-Whisper
This script helps install and verify faster-whisper installation.

Author: Geethika Isuru
License: MIT
"""

import subprocess
import sys
import importlib.util

def check_package_installed(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def test_faster_whisper():
    """Test faster-whisper installation."""
    try:
        from faster_whisper import WhisperModel
        print("✅ Faster-Whisper import successful!")
        
        # Test model loading
        print("Testing model loading...")
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Model loading successful!")
        
        # Test transcription
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        segments, info = model.transcribe(dummy_audio)
        list(segments)  # Consume generator
        print("✅ Transcription test successful!")
        
        return True
    except Exception as e:
        print(f"❌ Faster-Whisper test failed: {e}")
        return False

def main():
    print("🚀 Faster-Whisper Installation Helper")
    print("=" * 50)
    
    # Check if faster-whisper is already installed
    if check_package_installed("faster_whisper"):
        print("✅ Faster-Whisper is already installed!")
        
        # Test the installation
        if test_faster_whisper():
            print("\n🎉 Faster-Whisper is ready to use!")
        else:
            print("\n⚠️ Faster-Whisper is installed but not working properly.")
            print("Try reinstalling with: pip install --upgrade faster-whisper")
    else:
        print("❌ Faster-Whisper is not installed.")
        
        # Ask user if they want to install
        response = input("\nWould you like to install faster-whisper? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            success = install_package("faster-whisper")
            
            if success:
                print("\nTesting installation...")
                if test_faster_whisper():
                    print("\n🎉 Faster-Whisper installed and working correctly!")
                else:
                    print("\n⚠️ Installation completed but tests failed.")
                    print("You may need to restart your Python environment.")
            else:
                print("\n❌ Installation failed. Please try manually:")
                print("pip install faster-whisper")
        else:
            print("Installation cancelled.")
    
    print("\n📋 Additional Information:")
    print("- Faster-Whisper GitHub: https://github.com/SYSTRAN/faster-whisper")
    print("- For GPU acceleration, ensure you have CUDA installed")
    print("- Run the main application with: streamlit run streamlit_transcription.py")

if __name__ == "__main__":
    main() 