#!/usr/bin/env python3
"""
Streamlit Live Transcription with OpenAI Whisper and Faster-Whisper
Real-time speech recognition from microphone input using GPU acceleration.
Web-based interface with start/stop controls and keyboard shortcuts.
Optimized for NVIDIA RTX 4050 with 6GB VRAM.

Features:
- Support for both OpenAI Whisper and Faster-Whisper (CTranslate2)
- GPU acceleration with FP16 and INT8 quantization
- Real-time transcription with configurable chunk processing
- VAD (Voice Activity Detection) filtering
- Keyboard shortcuts and web interface controls

Controls:
- Start/Stop Button: Toggle listening
- SHIFT + SPACE: Start/Stop listening (keyboard shortcut)
- Clear button: Clear transcription history

Author: Geethika Isuru
GitHub: https://github.com/geethikaisuru
LinkedIn: https://www.linkedin.com/in/geethikaisuru/
Website: https://geethikaisuru.com

License: MIT License
"""

import streamlit as st
import pyaudio
import numpy as np
import torch
import threading
import time
import queue
import os
from collections import deque
import streamlit.components.v1 as components
import logging

# Import both whisper implementations
try:
    import whisper as openai_whisper
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Whisper Live",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitLiveTranscriber:
    def __init__(self, model_name="small", chunk_duration=2.0, overlap_duration=0.5, 
                 device=None, whisper_implementation="faster-whisper", compute_type="float16", 
                 use_vad=True):
        """
        Initialize the live transcriber for Streamlit.
        
        Args:
            model_name (str): Whisper model to use ("tiny", "base", "small", "medium", "large", "turbo", "distil-large-v3")
            chunk_duration (float): Duration of each audio chunk in seconds
            overlap_duration (float): Overlap between chunks to avoid cutting words
            device (str): Device to use ("cuda", "cpu", or None for auto-detection)
            whisper_implementation (str): "openai-whisper" or "faster-whisper"
            compute_type (str): For faster-whisper: "float16", "int8_float16", "int8"
            use_vad (bool): Use Voice Activity Detection to filter silence
        """
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.whisper_implementation = whisper_implementation
        self.compute_type = compute_type
        self.use_vad = use_vad
        
        # Audio settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap_size = int(self.sample_rate * self.overlap_duration)
        
        # Threading and buffering
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.running = False
        self.listening = False
        self.should_exit = False
        
        # Audio buffer for overlap
        self.audio_buffer = deque(maxlen=self.overlap_size)
        
        # Streamlit-specific attributes
        self.transcriptions = []
        self.status_message = "Ready to start transcription"
        self.device_info = ""
        
        # Initialize PyAudio
        self.audio = None
        self.stream = None
        self.audio_thread = None
        self.transcription_thread = None
        
        # Model loading state
        self.model = None
        self.device = None
        self.model_loaded = False
        
    def _determine_device(self, preferred_device=None):
        """Determine the best device to use for inference."""
        device_info = []
        device_info.append("🔍 Detecting available devices...")
        
        # Check PyTorch version
        device_info.append(f"📦 PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        device_info.append(f"🎮 CUDA available: {cuda_available}")
        
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                device_info.append(f"🚀 GPU: {gpu_name}")
                device_info.append(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                
                # Test CUDA functionality
                test_tensor = torch.tensor([1.0]).cuda()
                device_info.append("✅ CUDA test successful!")
                
                if preferred_device == "cpu":
                    device_info.append("⚙️ User requested CPU, using CPU")
                    self.device_info = "\n".join(device_info)
                    return "cpu"
                else:
                    device_info.append("🎯 Using GPU (CUDA) - Recommended for RTX 4050!")
                    self.device_info = "\n".join(device_info)
                    return "cuda"
                    
            except Exception as e:
                device_info.append(f"⚠️ CUDA error: {e}")
                device_info.append("🔄 Falling back to CPU")
                self.device_info = "\n".join(device_info)
                return "cpu"
        else:
            device_info.append("🔍 CUDA diagnostics:")
            
            # Check CUDA installation
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                device_info.append(f"  ✅ CUDA_PATH: {cuda_path}")
            else:
                device_info.append("  ❌ CUDA_PATH not set")
            
            # Check if we have the right PyTorch version
            if "+cu" in torch.__version__:
                device_info.append(f"  ✅ CUDA-enabled PyTorch: {torch.__version__}")
                device_info.append("  💡 CUDA might need system restart to work properly")
            else:
                device_info.append(f"  ❌ CPU-only PyTorch: {torch.__version__}")
            
            device_info.append("🖥️ Using CPU")
            self.device_info = "\n".join(device_info)
            return "cpu"
    
    def load_model(self, device_preference=None):
        """Load the Whisper model based on the selected implementation."""
        if self.model_loaded:
            return True
            
        try:
            # Determine device to use
            self.device = self._determine_device(device_preference)
            
            # Load model based on implementation
            if self.whisper_implementation == "faster-whisper":
                return self._load_faster_whisper_model()
            else:
                return self._load_openai_whisper_model()
                
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def _load_faster_whisper_model(self):
        """Load Faster-Whisper model."""
        try:
            with st.spinner(f"Loading Faster-Whisper {self.model_name} model..."):
                # Determine compute type based on device
                if self.device == "cuda":
                    compute_type = self.compute_type
                else:
                    compute_type = "int8" if self.compute_type in ["int8_float16", "int8"] else "float32"
                
                self.model = WhisperModel(
                    self.model_name, 
                    device=self.device, 
                    compute_type=compute_type
                )
                
            # Pre-warm the model
            with st.spinner("Pre-warming Faster-Whisper model..."):
                dummy_audio = np.zeros(self.sample_rate * 2, dtype=np.float32)
                segments, _ = self.model.transcribe(
                    dummy_audio, 
                    language="en",
                    vad_filter=False
                )
                # Consume the generator to actually run transcription
                list(segments)
                
            self.model_loaded = True
            self.status_message = f"✅ Faster-Whisper model loaded successfully on {self.device.upper()}! (Compute: {self.compute_type})"
            return True
            
        except Exception as e:
            if self.device == "cuda":
                st.warning("🔄 CUDA failed with Faster-Whisper, falling back to CPU...")
                try:
                    self.device = "cpu"
                    compute_type = "int8" if self.compute_type in ["int8_float16", "int8"] else "float32"
                    self.model = WhisperModel(self.model_name, device=self.device, compute_type=compute_type)
                    
                    # Pre-warm
                    dummy_audio = np.zeros(self.sample_rate * 2, dtype=np.float32)
                    segments, _ = self.model.transcribe(dummy_audio, language="en", vad_filter=False)
                    list(segments)
                    
                    self.model_loaded = True
                    self.status_message = f"✅ Faster-Whisper model loaded successfully on CPU! (Compute: {compute_type})"
                    return True
                except Exception as cpu_e:
                    st.error(f"❌ Error loading Faster-Whisper on CPU: {cpu_e}")
                    return False
            else:
                st.error(f"❌ Error loading Faster-Whisper: {e}")
                return False
    
    def _load_openai_whisper_model(self):
        """Load OpenAI Whisper model."""
        try:
            with st.spinner(f"Loading OpenAI Whisper {self.model_name} model..."):
                self.model = openai_whisper.load_model(self.model_name, device=self.device)
                
            # Pre-warm the model
            with st.spinner("Pre-warming OpenAI Whisper model..."):
                dummy_audio = np.zeros(self.sample_rate * 2, dtype=np.float32)
                _ = self.model.transcribe(dummy_audio, language="en", fp16=self.device=="cuda")
                
            self.model_loaded = True
            self.status_message = f"✅ OpenAI Whisper model loaded successfully on {self.device.upper()}!"
            return True
            
        except Exception as e:
            if self.device == "cuda":
                st.warning("🔄 CUDA failed with OpenAI Whisper, falling back to CPU...")
                try:
                    self.device = "cpu"
                    self.model = openai_whisper.load_model(self.model_name, device=self.device)
                    dummy_audio = np.zeros(self.sample_rate * 2, dtype=np.float32)
                    _ = self.model.transcribe(dummy_audio, language="en", fp16=False)
                    self.model_loaded = True
                    self.status_message = "✅ OpenAI Whisper model loaded successfully on CPU!"
                    return True
                except Exception as cpu_e:
                    st.error(f"❌ Error loading OpenAI Whisper on CPU: {cpu_e}")
                    return False
            else:
                st.error(f"❌ Error loading OpenAI Whisper: {e}")
                return False
    
    def get_audio_devices(self):
        """Get available audio input devices."""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
            
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels']
                })
        return devices
    
    def get_default_input_device(self):
        """Get the default input device."""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
            
        try:
            info = self.audio.get_default_input_device_info()
            if info['maxInputChannels'] > 0:
                return info['index']
        except:
            pass
        
        # Fallback to first available input device
        devices = self.get_audio_devices()
        if devices:
            return devices[0]['index']
        return None
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for audio stream."""
        if self.running and self.listening:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def audio_processor(self):
        """Process audio chunks and add them to transcription queue."""
        current_chunk = np.array([], dtype=np.float32)
        
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                current_chunk = np.concatenate([current_chunk, audio_data])
                
                if len(current_chunk) >= self.chunk_size:
                    # Add overlap from previous chunk
                    if len(self.audio_buffer) > 0:
                        chunk_with_overlap = np.concatenate([
                            np.array(self.audio_buffer), 
                            current_chunk[:self.chunk_size]
                        ])
                    else:
                        chunk_with_overlap = current_chunk[:self.chunk_size]
                    
                    # Store overlap for next chunk
                    self.audio_buffer.extend(current_chunk[self.chunk_size-self.overlap_size:self.chunk_size])
                    
                    # Add to transcription queue
                    self.transcription_queue.put(chunk_with_overlap)
                    
                    # Keep remaining data for next chunk
                    current_chunk = current_chunk[self.chunk_size:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    st.error(f"Error in audio processor: {e}")
    
    def _transcribe_faster_whisper(self, audio_chunk):
        """Transcribe using Faster-Whisper implementation."""
        segments, info = self.model.transcribe(
            audio_chunk,
            language="en",
            vad_filter=self.use_vad,
            beam_size=5
        )
        
        # Collect all segments
        text_segments = []
        for segment in segments:
            text_segments.append(segment.text.strip())
        
        return " ".join(text_segments), info.language_probability if hasattr(info, 'language_probability') else 0.0
    
    def _transcribe_openai_whisper(self, audio_chunk):
        """Transcribe using OpenAI Whisper implementation."""
        result = self.model.transcribe(
            audio_chunk,
            language="en",
            fp16=self.device=="cuda",
            verbose=False
        )
        
        return result["text"].strip(), result.get("language_probability", 0.0)
    
    def transcription_processor(self):
        """Process audio chunks and transcribe them."""
        while self.running:
            try:
                audio_chunk = self.transcription_queue.get(timeout=0.1)
                
                # Normalize audio
                audio_chunk = audio_chunk.astype(np.float32)
                if np.max(np.abs(audio_chunk)) > 0:
                    audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
                
                # Transcribe based on implementation
                start_time = time.time()
                
                if self.whisper_implementation == "faster-whisper":
                    text, confidence = self._transcribe_faster_whisper(audio_chunk)
                else:
                    text, confidence = self._transcribe_openai_whisper(audio_chunk)
                
                transcription_time = time.time() - start_time
                
                if text:
                    device_emoji = "🚀" if self.device == "cuda" else "🖥️"
                    impl_emoji = "⚡" if self.whisper_implementation == "faster-whisper" else "🤖"
                    timestamp = time.strftime("%H:%M:%S")
                    
                    # Add to transcriptions list
                    self.transcriptions.append({
                        'timestamp': timestamp,
                        'text': text,
                        'processing_time': transcription_time,
                        'device': device_emoji,
                        'implementation': impl_emoji,
                        'confidence': confidence
                    })
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    st.error(f"Error in transcription processor: {e}")
    
    def start_transcription(self, device_index=None):
        """Start the live transcription."""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
            
        if device_index is None:
            device_index = self.get_default_input_device()
        
        if device_index is None:
            st.error("No audio input device found!")
            return False
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            
            # Start processing threads
            self.running = True
            self.audio_thread = threading.Thread(target=self.audio_processor, daemon=True)
            self.transcription_thread = threading.Thread(target=self.transcription_processor, daemon=True)
            
            self.audio_thread.start()
            self.transcription_thread.start()
            
            # Start audio stream
            self.stream.start_stream()
            
            impl_name = "Faster-Whisper" if self.whisper_implementation == "faster-whisper" else "OpenAI Whisper"
            self.status_message = f"🎤 {impl_name} transcription system started - Ready to listen!"
            return True
            
        except Exception as e:
            st.error(f"❌ Error starting transcription: {e}")
            return False
    
    def toggle_listening(self):
        """Toggle listening state."""
        if not self.running:
            return False
            
        self.listening = not self.listening
        if self.listening:
            self.status_message = "🔴 LISTENING... Recording audio"
            # Clear any remaining audio data
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        else:
            self.status_message = "⏸️ PAUSED - Click Start or press SHIFT+SPACE to resume"
        
        return True
    
    def stop_transcription(self):
        """Stop the transcription."""
        self.running = False
        self.listening = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2)
            
        self.status_message = "⏹️ Transcription stopped"
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_transcription()
        if self.audio:
            self.audio.terminate()
            self.audio = None

# Initialize session state
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = None
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'system_running' not in st.session_state:
    st.session_state.system_running = False
if 'device_index' not in st.session_state:
    st.session_state.device_index = None

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🎤 Live Transcription with Whisper")
    st.markdown("Real-time speech recognition with GPU acceleration - **OpenAI Whisper** vs **Faster-Whisper**")
    
    # Check availability of implementations
    availability_cols = st.columns(2)
    with availability_cols[0]:
        if OPENAI_WHISPER_AVAILABLE:
            st.success("✅ OpenAI Whisper Available")
        else:
            st.error("❌ OpenAI Whisper Not Available")
    
    with availability_cols[1]:
        if FASTER_WHISPER_AVAILABLE:
            st.success("✅ Faster-Whisper Available")
        else:
            st.error("❌ Faster-Whisper Not Available")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Implementation selection
        available_implementations = []
        if FASTER_WHISPER_AVAILABLE:
            available_implementations.append("faster-whisper")
        if OPENAI_WHISPER_AVAILABLE:
            available_implementations.append("openai-whisper")
        
        if not available_implementations:
            st.error("❌ No Whisper implementations available! Please install whisper or faster-whisper.")
            return
        
        selected_implementation = st.selectbox(
            "🚀 Whisper Implementation",
            available_implementations,
            index=0,
            format_func=lambda x: {
                "faster-whisper": "⚡ Faster-Whisper (Recommended - Up to 4x faster)",
                "openai-whisper": "🤖 OpenAI Whisper (Original)"
            }[x],
            help="""
            **Faster-Whisper**: CTranslate2 implementation, up to 4x faster with lower memory usage
            **OpenAI Whisper**: Original implementation from OpenAI
            """
        )
        
        # Model selection (with different options for faster-whisper)
        if selected_implementation == "faster-whisper":
            model_options = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo", "distil-large-v3"]
            default_model = "small"
        else:
            model_options = ["tiny", "base", "small", "medium", "large"]
            default_model = "small"
        
        selected_model = st.selectbox(
            "🤖 Whisper Model",
            model_options,
            index=model_options.index(default_model),
            help="""
            **tiny**: Fastest, least accurate (~39 MB)
            **base**: Good balance (~74 MB)
            **small**: Good quality (~244 MB, recommended)
            **medium**: Higher quality (~769 MB)
            **large**: Best quality (~1550 MB)
            **large-v3**: Latest large model (faster-whisper only)
            **turbo**: Optimized balance (faster-whisper only)
            **distil-large-v3**: Distilled model, faster inference (faster-whisper only)
            """
        )
        
        # Device selection
        device_options = ["auto", "cuda", "cpu"]
        selected_device = st.selectbox(
            "🎯 Processing Device",
            device_options,
            index=0,
            help="""
            **auto**: Automatically detect best device
            **cuda**: Force GPU usage (if available)
            **cpu**: Force CPU usage
            """
        )
        
        # Faster-Whisper specific settings
        if selected_implementation == "faster-whisper":
            st.subheader("⚡ Faster-Whisper Settings")
            
            compute_type_options = ["float16", "int8_float16", "int8", "float32"]
            compute_type = st.selectbox(
                "🔢 Compute Type",
                compute_type_options,
                index=0,
                help="""
                **float16**: Best quality, GPU recommended
                **int8_float16**: Good balance, GPU + CPU
                **int8**: Fastest, lower memory usage
                **float32**: Highest precision, CPU fallback
                """
            )
            
            use_vad = st.checkbox(
                "🔊 Voice Activity Detection (VAD)",
                value=True,
                help="Filter out silence automatically using Silero VAD"
            )
        else:
            compute_type = "float16"
            use_vad = False
        
        # Audio settings
        st.subheader("🎧 Audio Settings")
        chunk_duration = st.slider("Chunk Duration (seconds)", 1.0, 5.0, 2.0, 0.5)
        overlap_duration = st.slider("Overlap Duration (seconds)", 0.1, 1.0, 0.5, 0.1)
        
        # Initialize transcriber
        if st.button("🔄 Initialize System") or st.session_state.transcriber is None:
            if st.session_state.transcriber:
                st.session_state.transcriber.cleanup()
                st.session_state.system_running = False
                st.session_state.is_listening = False
            
            device_pref = None if selected_device == "auto" else selected_device
            st.session_state.transcriber = StreamlitLiveTranscriber(
                model_name=selected_model,
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                device=device_pref,
                whisper_implementation=selected_implementation,
                compute_type=compute_type,
                use_vad=use_vad
            )
            
            if st.session_state.transcriber.load_model(device_pref):
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize system")
                return
            
    
    # Main content area
    if st.session_state.transcriber and st.session_state.transcriber.model_loaded:
        
        # Implementation info banner
        impl_name = "Faster-Whisper" if st.session_state.transcriber.whisper_implementation == "faster-whisper" else "OpenAI Whisper"
        impl_emoji = "⚡" if st.session_state.transcriber.whisper_implementation == "faster-whisper" else "🤖"
        
        st.info(f"{impl_emoji} **Using {impl_name}** with model **{selected_model}** on **{st.session_state.transcriber.device.upper() if st.session_state.transcriber.device else 'Unknown'}**")
        
        # Device info
        with st.expander("📊 System Information", expanded=False):
            if st.session_state.transcriber.device_info:
                st.text(st.session_state.transcriber.device_info)
            
            # Implementation details
            st.subheader("🔧 Configuration Details")
            config_data = {
                "Implementation": impl_name,
                "Model": selected_model,
                "Device": st.session_state.transcriber.device.upper() if st.session_state.transcriber.device else "Unknown",
                "Compute Type": compute_type if selected_implementation == "faster-whisper" else "float16/float32",
                "VAD Enabled": use_vad if selected_implementation == "faster-whisper" else "N/A",
                "Chunk Duration": f"{chunk_duration}s",
                "Overlap Duration": f"{overlap_duration}s"
            }
            
            for key, value in config_data.items():
                st.text(f"{key}: {value}")
            
            # Audio devices
            st.subheader("🎤 Available Audio Devices")
            devices = st.session_state.transcriber.get_audio_devices()
            device_df = []
            for device in devices:
                device_df.append({
                    "Index": device['index'],
                    "Name": device['name'],
                    "Channels": device['channels']
                })
            if device_df:
                st.dataframe(device_df, use_container_width=True)
            
            # Audio device selection
            device_names = [f"{d['index']}: {d['name']}" for d in devices]
            if device_names:
                # Use session state for device selection
                default_idx = 0
                if st.session_state.device_index is not None:
                    for i, device in enumerate(devices):
                        if device['index'] == st.session_state.device_index:
                            default_idx = i
                            break
                
                selected_device_idx = st.selectbox(
                    "Select Audio Device",
                    range(len(device_names)),
                    index=default_idx,
                    format_func=lambda x: device_names[x]
                )
                st.session_state.device_index = devices[selected_device_idx]['index']
            else:
                st.session_state.device_index = None
                st.warning("No audio input devices found")
        
        # Control buttons
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        
        with col1:
            if not st.session_state.system_running:
                if st.button("🚀 Start System", type="primary", use_container_width=True):
                    if st.session_state.transcriber.start_transcription(st.session_state.device_index):
                        st.session_state.system_running = True
                        st.rerun()
            else:
                if st.button("⏹️ Stop System", type="secondary", use_container_width=True):
                    st.session_state.transcriber.stop_transcription()
                    st.session_state.system_running = False
                    st.session_state.is_listening = False
                    st.rerun()
        
        with col2:
            if st.session_state.system_running:
                if not st.session_state.is_listening:
                    if st.button("🔴 Start Listening", type="primary", use_container_width=True, key="start_listening"):
                        if st.session_state.transcriber.toggle_listening():
                            st.session_state.is_listening = True
                            st.rerun()
                else:
                    if st.button("⏸️ Pause Listening", type="secondary", use_container_width=True, key="pause_listening"):
                        if st.session_state.transcriber.toggle_listening():
                            st.session_state.is_listening = False
                            st.rerun()
        
        with col3:
            if st.button("🗑️ Clear History", use_container_width=True):
                if st.session_state.transcriber:
                    st.session_state.transcriber.transcriptions = []
                    st.rerun()
        
        with col4:
            st.button("🔄 Refresh", use_container_width=True)
        
        # Status display
        if st.session_state.transcriber:
            st.info(f"**Status:** {st.session_state.transcriber.status_message}")
        
        # Performance comparison info
        if selected_implementation == "faster-whisper":
            st.success("⚡ **Performance Boost**: Faster-Whisper is up to 4x faster than OpenAI Whisper with lower memory usage!")
        
        # Keyboard shortcut info
        st.markdown("""
        ### ⌨️ Keyboard Shortcuts
        - **SHIFT + SPACE**: Toggle listening (when system is running)
        - Use the buttons above for full control
        """)
        
        # Enhanced keyboard shortcut handler with session state update
        keyboard_script = f"""
        <script>
        let isListening = {str(st.session_state.is_listening).lower()};
        let systemRunning = {str(st.session_state.system_running).lower()};
        
        document.addEventListener('keydown', function(event) {{
            if (event.shiftKey && event.code === 'Space') {{
                event.preventDefault();
                if (systemRunning) {{
                    const buttonText = isListening ? 'Pause Listening' : 'Start Listening';
                    const buttons = parent.document.querySelectorAll('button');
                    for (let button of buttons) {{
                        if (button.textContent.includes(buttonText)) {{
                            button.click();
                            isListening = !isListening;
                            break;
                        }}
                    }}
                }}
            }}
        }});
        </script>
        """
        
        components.html(keyboard_script, height=0)
        
        # Transcription display
        st.markdown("---")
        st.subheader("📝 Live Transcriptions")
        
        # Auto-refresh container for transcriptions
        transcription_container = st.container()
        
        if st.session_state.transcriber and st.session_state.transcriber.transcriptions:
            with transcription_container:
                # Show recent transcriptions (last 20)
                recent_transcriptions = st.session_state.transcriber.transcriptions[-20:]
                
                for i, trans in enumerate(reversed(recent_transcriptions)):
                    with st.container():
                        col1, col2 = st.columns([1, 6])
                        with col1:
                            st.text(f"{trans['device']}{trans['implementation']} {trans['timestamp']}")
                            st.caption(f"{trans['processing_time']:.2f}s")
                            if 'confidence' in trans and trans['confidence'] > 0:
                                st.caption(f"Conf: {trans['confidence']:.2f}")
                        with col2:
                            st.markdown(f"**{trans['text']}**")
                        
                        if i < len(recent_transcriptions) - 1:
                            st.divider()
        else:
            with transcription_container:
                st.info("No transcriptions yet. Start the system and begin listening to see results here.")
        
        # Auto-refresh when listening
        if st.session_state.is_listening:
            time.sleep(0.5)
            st.rerun()
            
    else:
        st.warning("Please initialize the system using the sidebar settings.")
        st.markdown("""
        ### 🚀 Getting Started
        1. Choose between **Faster-Whisper** (recommended) or **OpenAI Whisper**
        2. Select your preferred Whisper model in the sidebar
        3. Configure processing device (GPU recommended)
        4. Click "Initialize System"
        5. Start the system and begin listening!
        
        ### ⚡ Why Faster-Whisper?
        - **Up to 4x faster** than OpenAI Whisper
        - **Lower memory usage** - great for RTX 4050
        - **8-bit quantization** support for even better performance
        - **Voice Activity Detection** built-in
        - **Same accuracy** as original Whisper
        """)

    # Author section
    st.markdown("---")
    st.subheader("👨‍💻 About the Creator")
    st.markdown("""
    **Geethika Isuru**  
    AI Engineer & Entrepreneur
    
    🔗 **Connect with me:**
    - [LinkedIn](https://www.linkedin.com/in/geethikaisuru/)
    - [GitHub](https://github.com/geethikaisuru)
    - [Website](https://geethikaisuru.com)
    
    📄 **License:** MIT License \n
    ⚠️ **Note:** The underlying Whisper Models are made by OpenAI and are open source with MIT License. Faster-Whisper is by SYSTRAN. I only made the frontend to easily use these models.
    - [OpenAI Whisper](https://github.com/openai/whisper)
    - [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)         
    """)
        
    # GitHub star button
    st.markdown("""
    <a href="https://github.com/geethikaisuru" target="_blank">
        <img src="https://img.shields.io/github/stars/geethikaisuru?style=social" alt="GitHub">
    </a>
    """,unsafe_allow_html=True)

if __name__ == "__main__":
    main() 