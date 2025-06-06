# 🚀 Open Source Distribution Guide

This guide explains how to distribute the **Streamlit Live Transcription** project as an open-source solution with multiple deployment options.

## 📋 Table of Contents

1. [GitHub Repository Setup](#github-repository-setup)
2. [Docker Distribution](#docker-distribution)
3. [Python Package (PyPI)](#python-package-pypi)
4. [Cloud Deployment Options](#cloud-deployment-options)
5. [Local Installation Methods](#local-installation-methods)
6. [CI/CD Pipeline](#cicd-pipeline)

---

## 🏗️ GitHub Repository Setup

### 1. Repository Structure
```
whisper-live-transcription/
├── streamlit_transcription.py      # Main app
├── live_transcription.py          # Original CLI version
├── requirements_streamlit.txt      # Dependencies
├── requirements.txt               # Core dependencies
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── .dockerignore                  # Docker ignore file
├── .gitignore                     # Git ignore file
├── LICENSE                        # MIT License
├── README.md                      # Main documentation
├── README_Streamlit.md            # Streamlit docs
├── STREAMLIT_SETUP.md             # Quick setup
├── DISTRIBUTION_GUIDE.md          # This file
├── run_streamlit.py               # Python launcher
├── run_streamlit.bat              # Windows launcher
├── setup.py                       # Package setup
├── pyproject.toml                 # Modern package config
├── .github/
│   └── workflows/
│       ├── docker-build.yml       # Docker CI/CD
│       ├── python-test.yml        # Python testing
│       └── release.yml            # Release automation
└── docs/                          # Additional documentation
```

### 2. Essential Files for Open Source

#### `.gitignore`
```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/

# Model cache
.cache/
*.cache

# Audio files
*.wav
*.mp3
*.flac
```

---

## 🐳 Docker Distribution

### 1. Create Dockerfile

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_streamlit.txt .
RUN pip install --no-cache-dir -r requirements_streamlit.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "streamlit_transcription.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  whisper-transcription:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - /dev/snd:/dev/snd  # Audio device access
    devices:
      - /dev/snd  # Audio devices
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
    
  # Optional: NVIDIA GPU support
  whisper-transcription-gpu:
    build: .
    ports:
      - "8502:8501"
    volumes:
      - /dev/snd:/dev/snd
    devices:
      - /dev/snd
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

### 3. Docker Commands

```bash
# Build the image
docker build -t whisper-live-transcription .

# Run the container
docker run -p 8501:8501 --device /dev/snd whisper-live-transcription

# Using docker-compose
docker-compose up -d

# For GPU support (requires NVIDIA Docker)
docker-compose up whisper-transcription-gpu -d
```

---

## 📦 Python Package (PyPI)

### 1. Create setup.py

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements_streamlit.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="whisper-live-transcription",
    version="1.0.0",
    author="Geethika Isuru",
    author_email="your-email@example.com",
    description="Real-time speech transcription using OpenAI Whisper with Streamlit interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/geethikaisuru/whisper-live-transcription",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "whisper-live=streamlit_transcription:main",
            "whisper-cli=live_transcription:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.bat"],
    },
)
```

### 2. Create pyproject.toml (Modern approach)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "whisper-live-transcription"
version = "1.0.0"
description = "Real-time speech transcription using OpenAI Whisper with Streamlit interface"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "Geethika Isuru", email = "your-email@example.com"},
]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
requires-python = ">=3.8"
dependencies = [
    "streamlit>=1.28.0",
    "openai-whisper>=20231117",
    "torch>=1.9.0",
    "numpy>=1.21.0",
    "pyaudio>=0.2.11",
    "keyboard>=0.13.5",
]

[project.urls]
Homepage = "https://github.com/geethikaisuru/whisper-live-transcription"
Repository = "https://github.com/geethikaisuru/whisper-live-transcription.git"
Issues = "https://github.com/geethikaisuru/whisper-live-transcription/issues"

[project.scripts]
whisper-live = "streamlit_transcription:main"
whisper-cli = "live_transcription:main"
```

### 3. PyPI Publishing Commands

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# Upload to TestPyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

---

## ☁️ Cloud Deployment Options

### 1. Streamlit Cloud (Free)

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

**Pros:** Free, easy, automatic updates
**Cons:** Limited resources, no GPU support

### 2. Heroku

Create `Procfile`:
```
web: streamlit run streamlit_transcription.py --server.port=$PORT --server.address=0.0.0.0
```

Create `runtime.txt`:
```
python-3.9.19
```

Deploy commands:
```bash
heroku create your-app-name
git push heroku main
```

### 3. Google Cloud Run

```bash
# Build and deploy
gcloud run deploy whisper-transcription \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### 4. AWS ECS/Fargate

Use the Docker image with AWS ECS for scalable deployment.

### 5. DigitalOcean App Platform

Connect your GitHub repo and deploy directly from the web interface.

---

## 💻 Local Installation Methods

### Method 1: pip install (once published)
```bash
pip install whisper-live-transcription
whisper-live  # Start Streamlit version
whisper-cli   # Start CLI version
```

### Method 2: GitHub Clone
```bash
git clone https://github.com/geethikaisuru/whisper-live-transcription.git
cd whisper-live-transcription
pip install -r requirements_streamlit.txt
python run_streamlit.py
```

### Method 3: Docker
```bash
docker run -p 8501:8501 --device /dev/snd geethikaisuru/whisper-live-transcription
```

### Method 4: One-liner install script
Create `install.sh`:
```bash
#!/bin/bash
echo "Installing Whisper Live Transcription..."
git clone https://github.com/geethikaisuru/whisper-live-transcription.git
cd whisper-live-transcription
pip install -r requirements_streamlit.txt
echo "Installation complete! Run: python run_streamlit.py"
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Create `.github/workflows/ci-cd.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  release:
    types: [ published ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev python3-pyaudio
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements_streamlit.txt
        pip install pytest flake8
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: |
        pytest tests/ || echo "No tests found"

  docker:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build and push Docker image
      run: |
        docker build -t geethikaisuru/whisper-live-transcription:latest .
        docker build -t geethikaisuru/whisper-live-transcription:${{ github.sha }} .
        # Add docker push commands if you have Docker Hub setup

  pypi:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'release'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

---

## 📊 Distribution Comparison

| Method | Ease of Use | Cost | GPU Support | Scalability | Best For |
|--------|-------------|------|-------------|-------------|----------|
| GitHub Clone | ⭐⭐⭐ | Free | ✅ | ⭐ | Developers |
| Docker | ⭐⭐⭐⭐ | Free | ✅ | ⭐⭐⭐⭐ | Self-hosting |
| PyPI Package | ⭐⭐⭐⭐⭐ | Free | ✅ | ⭐⭐ | End users |
| Streamlit Cloud | ⭐⭐⭐⭐⭐ | Free | ❌ | ⭐⭐ | Demos |
| Cloud Deploy | ⭐⭐⭐ | $$ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Production |

---

## 🎯 Recommended Distribution Strategy

1. **Primary**: GitHub repository with good documentation
2. **Easy install**: PyPI package for pip install
3. **Containerized**: Docker for consistent deployment
4. **Demo**: Streamlit Cloud for public demo
5. **CI/CD**: GitHub Actions for automation

This multi-pronged approach ensures maximum accessibility for different user types and use cases! 