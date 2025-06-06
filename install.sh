#!/bin/bash

# Whisper Live Transcription - Quick Install Script
# Author: Geethika Isuru

set -e  # Exit on any error

echo "🎤 Whisper Live Transcription - Quick Installer"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    print_error "Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION or higher is required."
    exit 1
fi

print_status "Python $PYTHON_VERSION detected"

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Create installation directory
INSTALL_DIR="$HOME/whisper-live-transcription"
if [ -d "$INSTALL_DIR" ]; then
    print_warning "Directory $INSTALL_DIR already exists. Removing..."
    rm -rf "$INSTALL_DIR"
fi

print_info "Installing to: $INSTALL_DIR"

# Clone the repository
print_info "Cloning repository..."
git clone https://github.com/geethikaisuru/whisper-live-transcription.git "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3 first."
    exit 1
fi

# Install system dependencies (Linux/Ubuntu)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_info "Detected Linux. Checking for system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        print_info "Installing system dependencies with apt-get..."
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev python3-pyaudio
    elif command -v yum &> /dev/null; then
        print_info "Installing system dependencies with yum..."
        sudo yum install -y portaudio-devel python3-pyaudio
    elif command -v pacman &> /dev/null; then
        print_info "Installing system dependencies with pacman..."
        sudo pacman -S portaudio python-pyaudio
    else
        print_warning "Unable to automatically install system dependencies. Please install portaudio manually."
    fi
fi

# macOS dependencies
if [[ "$OSTYPE" == "darwin"* ]]; then
    print_info "Detected macOS. Checking for Homebrew..."
    if command -v brew &> /dev/null; then
        print_info "Installing system dependencies with Homebrew..."
        brew install portaudio
    else
        print_warning "Homebrew not found. Please install portaudio manually."
    fi
fi

# Create virtual environment
print_info "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_info "Installing Python dependencies..."
pip install -r requirements_streamlit.txt

print_status "Installation completed successfully!"

echo ""
echo "🎉 Whisper Live Transcription is now installed!"
echo ""
echo "📋 To run the application:"
echo "   cd $INSTALL_DIR"
echo "   source venv/bin/activate"
echo "   python run_streamlit.py"
echo ""
echo "🔗 Or use the quick commands:"
echo "   cd $INSTALL_DIR && source venv/bin/activate && python run_streamlit.py"
echo ""
echo "📚 For more information, see:"
echo "   - README_Streamlit.md"
echo "   - STREAMLIT_SETUP.md"
echo ""
echo "👨‍💻 Created by Geethika Isuru"
echo "   LinkedIn: https://www.linkedin.com/in/geethikaisuru/"
echo "   GitHub: https://github.com/geethikaisuru"
echo "   Website: https://geethikaisuru.com" 