#!/usr/bin/env python3
"""
Live Transcription with OpenAI Whisper
Real-time speech recognition from microphone input using GPU acceleration.
Push-to-talk functionality with hotkey controls.
Optimized for NVIDIA RTX 4050 with 6GB VRAM.

Controls:
- SHIFT + SPACE: Start/Stop listening
- ESC: Quit application
"""

import whisper
import pyaudio
import numpy as np
import torch
import threading
import time
import queue
import sys
import keyboard
import os
from collections import deque

class LiveTranscriber:
    def __init__(self, model_name="small", chunk_duration=2.0, overlap_duration=0.5, device=None):
        """
        Initialize the live transcriber.
        
        Args:
            model_name (str): Whisper model to use ("tiny", "base", "small", "medium", "large", "turbo")
            chunk_duration (float): Duration of each audio chunk in seconds
            overlap_duration (float): Overlap between chunks to avoid cutting words
            device (str): Device to use ("cuda", "cpu", or None for auto-detection)
        """
        self.model_name = model_name
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        
        # Audio settings
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.channels = 1
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.overlap_size = int(self.sample_rate * self.overlap_duration)
        
        # Threading and buffering
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.running = False
        self.listening = False  # Controls whether we're actively recording
        self.should_exit = False
        
        # Audio buffer for overlap
        self.audio_buffer = deque(maxlen=self.overlap_size)
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Determine device to use
        self.device = self._determine_device(device)
        
        # Load Whisper model
        print(f"Loading Whisper {model_name} model...")
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            print(f"✅ Model loaded successfully on {self.device.upper()}!")
        except Exception as e:
            print(f"❌ Error loading model on {self.device}: {e}")
            if self.device == "cuda":
                print("🔄 Falling back to CPU...")
                self.device = "cpu"
                self.model = whisper.load_model(model_name, device=self.device)
                print("✅ Model loaded successfully on CPU!")
            else:
                raise e
        
        # Pre-warm the model
        print("🔥 Pre-warming model...")
        try:
            dummy_audio = np.zeros(self.sample_rate * 2, dtype=np.float32)
            _ = self.model.transcribe(dummy_audio, language="en", fp16=self.device=="cuda")
            print("✅ Model pre-warmed!")
        except Exception as e:
            print(f"⚠️ Pre-warming warning: {e}")
    
    def _determine_device(self, preferred_device=None):
        """Determine the best device to use for inference."""
        print("\n🔍 Detecting available devices...")
        
        # Check PyTorch version
        print(f"📦 PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"🎮 CUDA available: {cuda_available}")
        
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"🚀 GPU: {gpu_name}")
                print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
                
                # Test CUDA functionality
                test_tensor = torch.tensor([1.0]).cuda()
                print("✅ CUDA test successful!")
                
                if preferred_device == "cpu":
                    print("⚙️ User requested CPU, using CPU")
                    return "cpu"
                else:
                    print("🎯 Using GPU (CUDA) - Recommended for RTX 4050!")
                    return "cuda"
                    
            except Exception as e:
                print(f"⚠️ CUDA error: {e}")
                print("🔄 Falling back to CPU")
                return "cpu"
        else:
            # Check why CUDA might not be available
            print("🔍 CUDA diagnostics:")
            
            # Check if NVIDIA driver is available
            try:
                os.system("nvidia-smi --query-gpu=name --format=csv,noheader,nounits > nul 2>&1")
                print("  ✅ NVIDIA driver detected")
            except:
                print("  ❌ NVIDIA driver not found")
            
            # Check CUDA installation
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                print(f"  ✅ CUDA_PATH: {cuda_path}")
            else:
                print("  ❌ CUDA_PATH not set")
            
            # Check if we have the right PyTorch version
            if "+cu" in torch.__version__:
                print(f"  ✅ CUDA-enabled PyTorch: {torch.__version__}")
                print("  💡 CUDA might need system restart to work properly")
            else:
                print(f"  ❌ CPU-only PyTorch: {torch.__version__}")
            
            print("🖥️ Using CPU")
            return "cpu"
    
    def list_audio_devices(self):
        """List available audio input devices."""
        print("\n🎤 Available audio input devices:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  {i}: {info['name']} (Channels: {info['maxInputChannels']})")
    
    def get_default_input_device(self):
        """Get the default input device."""
        try:
            info = self.audio.get_default_input_device_info()
            if info['maxInputChannels'] > 0:
                return info['index']
        except:
            pass
        
        # Fallback to first available input device
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                return i
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
                # Get audio data with timeout
                audio_data = self.audio_queue.get(timeout=0.1)
                current_chunk = np.concatenate([current_chunk, audio_data])
                
                # Check if we have enough data for a chunk
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
                if self.running:  # Only print error if we're supposed to be running
                    print(f"Error in audio processor: {e}")
    
    def transcription_processor(self):
        """Process audio chunks and transcribe them."""
        while self.running:
            try:
                # Get audio chunk with timeout
                audio_chunk = self.transcription_queue.get(timeout=0.1)
                
                # Normalize audio
                audio_chunk = audio_chunk.astype(np.float32)
                if np.max(np.abs(audio_chunk)) > 0:
                    audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
                
                # Transcribe
                start_time = time.time()
                result = self.model.transcribe(
                    audio_chunk,
                    language="en",  # Set to None for auto-detection
                    fp16=self.device=="cuda",
                    verbose=False
                )
                transcription_time = time.time() - start_time
                
                text = result["text"].strip()
                if text:  # Only print non-empty transcriptions
                    device_emoji = "🚀" if self.device == "cuda" else "🖥️"
                    print(f"{device_emoji} [{transcription_time:.2f}s] {text}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:  # Only print error if we're supposed to be running
                    print(f"Error in transcription processor: {e}")
    
    def toggle_listening(self):
        """Toggle listening state."""
        self.listening = not self.listening
        if self.listening:
            print("🔴 LISTENING... (Press SHIFT+SPACE to stop)")
            # Clear any remaining audio data
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
        else:
            print("⏸️  STOPPED listening (Press SHIFT+SPACE to start)")
    
    def setup_hotkeys(self):
        """Setup keyboard hotkeys."""
        try:
            # SHIFT + SPACE to toggle listening
            keyboard.add_hotkey('shift+space', self.toggle_listening, suppress=True)
            
            # ESC to quit
            keyboard.add_hotkey('esc', self.stop_transcription, suppress=True)
            
            print("✅ Hotkeys registered successfully!")
            print("   SHIFT + SPACE: Start/Stop listening")
            print("   ESC: Quit application")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not setup hotkeys: {e}")
            print("You may need to run as administrator for global hotkeys to work.")
            return False
        return True
    
    def stop_transcription(self):
        """Stop the transcription."""
        print("\n🛑 Stopping transcription...")
        self.should_exit = True
        self.running = False
        self.listening = False
    
    def start(self, device_index=None):
        """Start the live transcription."""
        if device_index is None:
            device_index = self.get_default_input_device()
        
        if device_index is None:
            print("No audio input device found!")
            return
        
        # Validate device is an input device
        device_info = self.audio.get_device_info_by_index(device_index)
        if device_info['maxInputChannels'] == 0:
            print(f"Device {device_index} is not an input device. Finding alternative...")
            device_index = self.get_default_input_device()
            if device_index is None:
                print("No valid audio input device found!")
                return
            device_info = self.audio.get_device_info_by_index(device_index)
        
        print(f"\n🎧 Using audio device: {device_info['name']}")
        print(f"📊 Sample rate: {self.sample_rate} Hz")
        print(f"⏱️  Chunk duration: {self.chunk_duration} seconds")
        print(f"🤖 Model: {self.model_name}")
        print(f"🎯 Processing device: {self.device.upper()}")
        print("\n" + "="*60)
        
        # Setup hotkeys
        hotkeys_ok = self.setup_hotkeys()
        
        print("\n⏸️  Ready! Press SHIFT+SPACE to start listening...")
        print("="*60)
        
        try:
            # Open audio stream
            stream = self.audio.open(
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
            audio_thread = threading.Thread(target=self.audio_processor, daemon=True)
            transcription_thread = threading.Thread(target=self.transcription_processor, daemon=True)
            
            audio_thread.start()
            transcription_thread.start()
            
            # Start audio stream
            stream.start_stream()
            
            # Keep running until ESC is pressed
            try:
                while self.running and not self.should_exit:
                    time.sleep(0.1)
                    
                    # Alternative: Check for manual keyboard input if hotkeys don't work
                    if not hotkeys_ok:
                        if keyboard.is_pressed('space') and keyboard.is_pressed('shift'):
                            if not hasattr(self, '_last_toggle') or time.time() - self._last_toggle > 0.5:
                                self.toggle_listening()
                                self._last_toggle = time.time()
                        elif keyboard.is_pressed('esc'):
                            break
                            
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by Ctrl+C")
            
            # Cleanup
            self.running = False
            self.listening = False
            
            print("🧹 Cleaning up...")
            stream.stop_stream()
            stream.close()
            
            if hotkeys_ok:
                keyboard.unhook_all_hotkeys()
            
            # Wait for threads to finish
            if audio_thread.is_alive():
                audio_thread.join(timeout=2)
            if transcription_thread.is_alive():
                transcription_thread.join(timeout=2)
                
            print("✅ Transcription stopped successfully!")
            
        except Exception as e:
            print(f"❌ Error starting transcription: {e}")
        finally:
            if hasattr(self, 'audio'):
                self.audio.terminate()
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'audio'):
            self.audio.terminate()

def main():
    """Main function."""
    print("🎤 OpenAI Whisper Live Transcription with GPU Support")
    print("=" * 55)
    
    # Check if running as administrator (for global hotkeys)
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            print("⚠️  Note: For best hotkey support, consider running as administrator")
    except:
        pass
    
    # Device selection
    print("\nDevice options:")
    print("  auto: Automatically detect best device (GPU preferred)")
    print("  cuda: Force GPU usage")
    print("  cpu: Force CPU usage")
    device_choice = input("Enter device preference (or press Enter for 'auto'): ").strip().lower()
    if device_choice not in ['cuda', 'cpu']:
        device_choice = None  # Auto-detect
    
    # Model selection
    print("\nAvailable models:")
    print("  tiny:   Fastest, least accurate")
    print("  base:   Good balance")
    print("  small:  Good quality (recommended)")
    print("  medium: Higher quality (good for RTX 4050)")
    print("  large:  Best quality")
    print("  turbo:  Optimized balance of speed and quality")
    
    model_choice = input("Enter model name (or press Enter for 'medium'): ").strip().lower()
    if not model_choice:
        model_choice = "medium"  # Default to medium for GPU usage
    
    transcriber = LiveTranscriber(model_name=model_choice, device=device_choice)
    
    # List available devices
    transcriber.list_audio_devices()
    
    # Option to select device
    try:
        print("\nNote: Please select a microphone device (input device)")
        device_choice = input("Enter device number (or press Enter for default): ").strip()
        device_index = int(device_choice) if device_choice else None
    except (ValueError, KeyboardInterrupt):
        device_index = None
        print("Using default audio input device...")
    
    # Start transcription
    transcriber.start(device_index)

if __name__ == "__main__":
    main() 