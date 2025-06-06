# 🎤 Streamlit Live Transcription with Whisper

A web-based real-time speech recognition application using **OpenAI Whisper** and **Faster-Whisper** with GPU acceleration support. This Streamlit version provides an easy-to-use web interface with support for both implementations.

## ✨ Features

### 🚀 Dual Implementation Support
- **Faster-Whisper (Recommended)**: Up to 4x faster with CTranslate2 optimization
- **OpenAI Whisper**: Original implementation for compatibility

### 🎯 Core Features
- **Real-time transcription** with multiple Whisper models
- **GPU acceleration** support (CUDA) with FP16 and INT8 quantization
- **Web-based interface** - No command line needed
- **Voice Activity Detection (VAD)** - Automatic silence filtering (Faster-Whisper)
- **Multiple model options** including distil-large-v3
- **Audio device selection** - Choose your preferred microphone
- **Keyboard shortcuts** - SHIFT + SPACE to toggle listening
- **Live status updates** and transcription history
- **Automatic fallback** from GPU to CPU if needed
- **Device diagnostics** to troubleshoot GPU issues

## 🚀 Quick Start

### 1. Installation

```bash
# Install all dependencies (includes both implementations)
pip install -r requirements_streamlit.txt

# Or install manually
pip install streamlit openai-whisper faster-whisper pyaudio

# For CUDA support (optional, for GPU acceleration)
# Visit: https://pytorch.org/get-started/locally/
# Install PyTorch with CUDA support for your system
```

### 2. Test Faster-Whisper Installation (Optional)

```bash
# Run the installation helper
python install_faster_whisper.py
```

### 3. Run the Application

```bash
streamlit run streamlit_transcription.py
```

The application will open in your default web browser at `http://localhost:8501`

### 4. Usage

1. **Choose Implementation**: Select between Faster-Whisper (recommended) or OpenAI Whisper
2. **Configure Settings**: Choose model, device, and other preferences in the sidebar
3. **Initialize System**: Click "Initialize System" to load the model
4. **Start System**: Click "Start System" to begin audio capture
5. **Start Listening**: Click "Start Listening" or press SHIFT + SPACE to begin transcription
6. **View Results**: Real-time transcriptions appear with timestamps and processing times

## ⚙️ Configuration Options

### 🚀 Implementation Selection

#### Faster-Whisper (Recommended)
- **Up to 4x faster** than OpenAI Whisper
- **Lower memory usage** - Perfect for RTX 4050
- **8-bit quantization** support for even better performance
- **Built-in VAD** (Voice Activity Detection)
- **Additional models**: large-v2, large-v3, turbo, distil-large-v3

#### OpenAI Whisper (Original)
- **Original implementation** from OpenAI
- **Full compatibility** with existing workflows
- **Stable and tested** for all use cases

### Whisper Models

#### Standard Models (Both Implementations)
- **tiny**: Fastest, least accurate (~39MB)
- **base**: Good balance (~74MB)
- **small**: Good quality, recommended (~244MB)
- **medium**: Higher quality, good for RTX 4050 (~769MB)
- **large**: Best quality (~1550MB)

#### Faster-Whisper Exclusive Models
- **large-v2**: Improved large model
- **large-v3**: Latest large model with better accuracy
- **turbo**: Optimized balance of speed and quality
- **distil-large-v3**: Distilled model, 6x faster than large-v3

### Compute Types (Faster-Whisper Only)
- **float16**: Best quality, GPU recommended
- **int8_float16**: Good balance, works on GPU + CPU
- **int8**: Fastest, lowest memory usage
- **float32**: Highest precision, CPU fallback

### Processing Devices
- **auto**: Automatically detect best device (GPU preferred)
- **cuda**: Force GPU usage (requires CUDA-compatible GPU)
- **cpu**: Force CPU usage

### Audio Settings
- **Chunk Duration**: Length of audio segments for processing (1-5 seconds)
- **Overlap Duration**: Overlap between chunks to avoid cutting words (0.1-1 second)
- **VAD (Voice Activity Detection)**: Automatically filter silence (Faster-Whisper only)

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
- For Faster-Whisper: Models download from Hugging Face Hub
- Clear model cache if needed

#### 4. Import errors for Faster-Whisper
- Run the installation helper: `python install_faster_whisper.py`
- Install manually: `pip install faster-whisper`
- Check compatibility with your system

#### 5. Audio permission errors
- Grant microphone permissions to your browser
- On Windows: Check Windows privacy settings for microphone access
- Run browser as administrator if needed

#### 6. Keyboard shortcuts not working
- Ensure the Streamlit app has focus in the browser
- Try clicking on the page first
- Use the web interface buttons as alternative

### Performance Tips

1. **For RTX 4050 (6GB VRAM) - Recommended Setup**:
   - Use **Faster-Whisper** implementation
   - Use **medium** or **large-v3** model
   - Set compute type to **float16** or **int8_float16**
   - Enable **VAD** for better performance
   - Set chunk duration to 2-3 seconds

2. **For lower-end systems**:
   - Use **Faster-Whisper** with **int8** quantization
   - Use **small** or **distil-large-v3** model
   - Use CPU processing if needed
   - Increase chunk duration to reduce processing frequency

3. **For best accuracy**:
   - Use **large-v3** model (Faster-Whisper)
   - Use **float16** compute type
   - Ensure good microphone quality
   - Minimize background noise
   - Enable VAD to filter silence

## 📊 Model Comparison

### Standard Models (Both Implementations)

| Model  | Size   | OpenAI Whisper | Faster-Whisper | VRAM Usage | Recommended For |
|--------|--------|----------------|-----------------|------------|-----------------|
| tiny   | 39MB   | Baseline       | 4x faster      | <1GB       | Quick testing   |
| base   | 74MB   | Baseline       | 4x faster      | ~1GB       | Basic use       |
| small  | 244MB  | Baseline       | 4x faster      | ~2GB       | General use     |
| medium | 769MB  | Baseline       | 4x faster      | ~4GB       | RTX 4050        |
| large  | 1550MB | Baseline       | 4x faster      | ~8GB       | High-end GPUs   |

### Faster-Whisper Exclusive Models

| Model           | Size   | Speed vs Large | Accuracy | VRAM Usage | Description |
|-----------------|--------|----------------|----------|------------|-------------|
| large-v2        | 1550MB | Same          | Better   | ~8GB       | Improved large |
| large-v3        | 1550MB | Same          | Best     | ~8GB       | Latest model |
| turbo           | 809MB  | 8x faster     | Excellent| ~4GB       | Balanced option |
| distil-large-v3 | 756MB  | 6x faster     | Near-large| ~4GB      | Distilled model |

## 🔄 Differences from Original Script

The Streamlit version maintains all core functionality while adding:

- **Dual implementation support**: Choose between Faster-Whisper and OpenAI Whisper
- **Web-based interface**: No command line needed
- **Advanced configuration**: Quantization, VAD, and more options
- **Visual feedback**: Status indicators and progress bars
- **Device selection UI**: Easy audio device selection
- **Live updates**: Real-time transcription display
- **History management**: View and clear transcription history
- **Session persistence**: Settings maintained during session
- **Error handling**: Better error messages and recovery
- **Performance monitoring**: Processing time and confidence scores

## 🏆 Why Choose Faster-Whisper?

Based on the [official benchmarks](https://github.com/SYSTRAN/faster-whisper):

1. **4x Faster Processing**: Same accuracy, much faster speed
2. **Lower Memory Usage**: Better for systems with limited VRAM
3. **Quantization Support**: INT8 and FP16 for even better performance
4. **Built-in VAD**: Automatic silence detection and filtering
5. **Additional Models**: Access to latest and optimized models
6. **Better Resource Efficiency**: Perfect for RTX 4050 and similar GPUs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for the original Whisper model
- **SYSTRAN** for the Faster-Whisper implementation
- **CTranslate2** team for the optimization framework

## 👨‍💻 About the Creator

Hi! I'm **Geethika Isuru**, an AI Engineer & Entrepreneur who's trying to make a better world with AI.

- 💼 [LinkedIn Profile](https://www.linkedin.com/in/geethikaisuru/)
- 📂 [GitHub Profile](https://github.com/geethikaisuru)
- 🖥️ [Official Website](https://geethikaisuru.com)

## 🤝 Contributing

Feel free to submit issues and enhancement requests! 