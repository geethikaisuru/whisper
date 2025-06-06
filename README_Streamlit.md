# 🎤 Streamlit Live Transcription with OpenAI Whisper

A web-based real-time speech recognition application using OpenAI Whisper with GPU acceleration support. This Streamlit version provides an easy-to-use web interface for the original command-line transcription tool.

## ✨ Features

- **Real-time transcription** with OpenAI Whisper models
- **GPU acceleration** support (CUDA) for faster processing
- **Web-based interface** - No command line needed
- **Multiple Whisper models** (tiny, base, small, medium, large, turbo)
- **Audio device selection** - Choose your preferred microphone
- **Keyboard shortcuts** - SHIFT + SPACE to toggle listening
- **Live status updates** and transcription history
- **Automatic fallback** from GPU to CPU if needed
- **Device diagnostics** to troubleshoot GPU issues

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# For CUDA support (optional, for GPU acceleration)
# Visit: https://pytorch.org/get-started/locally/
# Install PyTorch with CUDA support for your system
```

### 2. Run the Application

```bash
streamlit run streamlit_transcription.py
```

The application will open in your default web browser at `http://localhost:8501`

### 3. Usage

1. **Initialize System**: Choose your model and device preferences in the sidebar, then click "Initialize System"
2. **Start System**: Click "Start System" to begin audio capture
3. **Start Listening**: Click "Start Listening" or press SHIFT + SPACE to begin transcription
4. **View Results**: Real-time transcriptions appear in the main area with timestamps and processing times

## ⚙️ Configuration Options

### Whisper Models
- **tiny**: Fastest, least accurate (~32MB)
- **base**: Good balance (~74MB)
- **small**: Good quality, recommended (~244MB)
- **medium**: Higher quality, good for RTX 4050 (~769MB)
- **large**: Best quality (~1550MB)
- **turbo**: Optimized balance of speed and quality (~809MB)

### Processing Devices
- **auto**: Automatically detect best device (GPU preferred)
- **cuda**: Force GPU usage (requires CUDA-compatible GPU)
- **cpu**: Force CPU usage

### Audio Settings
- **Chunk Duration**: Length of audio segments for processing (1-5 seconds)
- **Overlap Duration**: Overlap between chunks to avoid cutting words (0.1-1 second)

## 🎮 Controls

### Web Interface
- **🚀 Start System**: Initialize audio capture
- **🔴 Start Listening**: Begin recording and transcription
- **⏸️ Pause Listening**: Pause transcription (system remains active)
- **⏹️ Stop System**: Stop all processes
- **🗑️ Clear History**: Clear transcription history
- **🔄 Refresh**: Refresh the page

### Keyboard Shortcuts
- **SHIFT + SPACE**: Toggle listening state (when system is running)

## 🔧 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- Microphone input
- Internet connection (for initial model download)

### Recommended for GPU Acceleration
- NVIDIA GPU with CUDA support
- 6GB+ VRAM (for medium/large models)
- CUDA Toolkit installed
- PyTorch with CUDA support

### Audio Requirements
- Working microphone
- Audio input permissions for the browser/system

## 🛠️ Troubleshooting

### Common Issues

#### 1. "No audio input device found"
- Check microphone permissions
- Ensure microphone is connected and working
- Try selecting a different audio device in System Information

#### 2. CUDA not available
- Install NVIDIA drivers
- Install CUDA Toolkit
- Install PyTorch with CUDA support:
  ```bash
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

#### 3. Model loading errors
- Check internet connection (models download on first use)
- Try a smaller model (tiny/base) first
- Clear model cache: Delete `~/.cache/whisper/`

#### 4. Audio permission errors
- Grant microphone permissions to your browser
- On Windows: Check Windows privacy settings for microphone access
- Run browser as administrator if needed

#### 5. Keyboard shortcuts not working
- Ensure the Streamlit app has focus in the browser
- Try clicking on the page first
- Use the web interface buttons as alternative

### Performance Tips

1. **For RTX 4050 (6GB VRAM)**:
   - Use "medium" model for best balance
   - Enable GPU acceleration
   - Set chunk duration to 2-3 seconds

2. **For lower-end systems**:
   - Use "small" or "base" model
   - Use CPU processing
   - Increase chunk duration to reduce processing frequency

3. **For best accuracy**:
   - Use "large" model (if system supports it)
   - Ensure good microphone quality
   - Minimize background noise

## 📊 Model Comparison

| Model  | Size  | Speed | Accuracy | VRAM Usage | Recommended For |
|--------|-------|-------|----------|------------|-----------------|
| tiny   | 32MB  | Very Fast | Basic | <1GB | Quick testing |
| base   | 74MB  | Fast | Good | ~1GB | Basic transcription |
| small  | 244MB | Medium | Very Good | ~2GB | General use |
| medium | 769MB | Slower | Excellent | ~4GB | RTX 4050 |
| large  | 1550MB| Slow | Best | ~8GB | High-end GPUs |
| turbo  | 809MB | Fast | Excellent | ~4GB | Balanced option |

## 🔄 Differences from Original Script

The Streamlit version maintains all core functionality while adding:

- **Web-based interface**: No command line needed
- **Visual feedback**: Status indicators and progress bars
- **Device selection UI**: Easy audio device selection
- **Live updates**: Real-time transcription display
- **History management**: View and clear transcription history
- **Session persistence**: Settings maintained during session
- **Error handling**: Better error messages and recovery

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 About the Creator

Hi! I'm **Geethika Isuru**, an AI Engineer & Entrepreneur who's trying to make a better world with AI.

- 💼 [LinkedIn Profile](https://www.linkedin.com/in/geethikaisuru/)
- 📂 [GitHub Profile](https://github.com/geethikaisuru)
- 🖥️ [Official Website](https://geethikaisuru.com)

## 🤝 Contributing

Feel free to submit issues and enhancement requests! 