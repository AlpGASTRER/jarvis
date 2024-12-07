"""
Text-to-Speech Processing Module

This module provides text-to-speech (TTS) functionality for the Jarvis AI Assistant.
It uses pyttsx3 for speech synthesis and supports various output formats including
raw audio bytes, base64 encoded audio, and streaming audio.

Key Features:
- Lazy initialization of TTS engine
- Thread-safe operations
- Multiple output formats (raw bytes, base64)
- Streaming support
- Configurable voice settings
- Error handling and retry mechanism

Dependencies:
- pyttsx3: For text-to-speech synthesis
- wave: For WAV file handling
- threading: For thread safety
- queue: For streaming support
"""

import pyttsx3
import io
import wave
import base64
import threading
from typing import Optional
import queue
import time

class TTSProcessor:
    """
    A thread-safe text-to-speech processor using pyttsx3.
    
    This class provides methods for converting text to speech with various output
    formats and configurations. It uses lazy initialization to create the TTS
    engine only when needed and ensures thread safety using locks.
    
    Attributes:
        _engine: pyttsx3 engine instance (lazy initialized)
        _engine_lock: Thread lock for engine operations
        _audio_queue: Queue for streaming audio data
        _sample_width: Audio sample width in bytes (2 for 16-bit)
        _sample_rate: Audio sample rate in Hz
        _channels: Number of audio channels (1 for mono)
    """
    
    def __init__(self):
        """Initialize the TTS processor with default settings."""
        self._engine = None
        self._engine_lock = threading.Lock()
        self._audio_queue = queue.Queue()
        self._sample_width = 2  # 16-bit audio
        self._sample_rate = 22050  # Standard TTS sample rate
        self._channels = 1  # Mono audio
        
    def _lazy_init_engine(self):
        """
        Initialize the TTS engine if not already initialized.
        
        Uses double-check locking pattern for thread safety.
        Configures default voice settings for neutral, clear speech.
        """
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:  # Double-check pattern
                    self._engine = pyttsx3.init()
                    self._engine.setProperty('rate', 175)  # Words per minute
                    self._engine.setProperty('volume', 1.0)  # Full volume
                    
                    # Configure voice (prefer neutral voice)
                    voices = self._engine.getProperty('voices')
                    for voice in voices:
                        if 'david' in voice.name.lower() or 'mark' in voice.name.lower():
                            self._engine.setProperty('voice', voice.id)
                            break
    
    def text_to_speech_base64(self, text: str, retry_count: int = 3) -> dict:
        """
        Convert text to speech and return as base64 encoded audio.
        
        Args:
            text: Text to convert to speech
            retry_count: Number of retries on failure (default: 3)
            
        Returns:
            dict: Result containing:
                - success: Boolean indicating success
                - audio_base64: Base64 encoded WAV audio (if successful)
                - sample_rate: Audio sample rate
                - channels: Number of audio channels
                - sample_width: Audio sample width
                - error: Error message (if failed)
        """
        try:
            self._lazy_init_engine()
            
            # Create an in-memory buffer for the audio
            buffer = io.BytesIO()
            
            # Configure the wave file
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(self._channels)
                wav_file.setsampwidth(self._sample_width)
                wav_file.setframerate(self._sample_rate)
                
                def callback(data):
                    wav_file.writeframes(data)
                
                # Set up the callback for audio data
                self._engine.connect('data', callback)
                
                # Generate speech with retry mechanism
                for attempt in range(retry_count):
                    try:
                        with self._engine_lock:
                            self._engine.say(text)
                            self._engine.runAndWait()
                        break
                    except RuntimeError as e:
                        if attempt == retry_count - 1:  # Last attempt
                            raise
                        time.sleep(0.1)  # Brief pause before retry
                        
            # Convert audio data to base64
            audio_data = buffer.getvalue()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            return {
                "success": True,
                "audio_base64": audio_base64,
                "sample_rate": self._sample_rate,
                "channels": self._channels,
                "sample_width": self._sample_width
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech and return raw audio bytes.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bytes: Raw WAV audio data, or None if failed
        """
        try:
            self._lazy_init_engine()
            
            # Create an in-memory buffer for the audio
            buffer = io.BytesIO()
            
            # Configure the wave file
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(self._channels)
                wav_file.setsampwidth(self._sample_width)
                wav_file.setframerate(self._sample_rate)
                
                def callback(data):
                    wav_file.writeframes(data)
                
                # Set up the callback for audio data
                self._engine.connect('data', callback)
                
                # Generate speech
                with self._engine_lock:
                    self._engine.say(text)
                    self._engine.runAndWait()
                        
            # Return the raw audio data
            return buffer.getvalue()
            
        except Exception as e:
            print(f"TTS error in processor: {str(e)}")
            return None
            
    def update_settings(self, settings: dict):
        """
        Update TTS engine settings.
        
        Args:
            settings: Dictionary containing settings to update:
                - rate: Speech rate in words per minute
                - volume: Volume level (0.0 to 1.0)
                - voice: Voice name to use
        """
        try:
            self._lazy_init_engine()
            with self._engine_lock:
                if "rate" in settings:
                    self._engine.setProperty('rate', settings["rate"])
                if "volume" in settings:
                    self._engine.setProperty('volume', settings["volume"])
                if "voice" in settings:
                    voices = self._engine.getProperty('voices')
                    for voice in voices:
                        if settings["voice"].lower() in voice.name.lower():
                            self._engine.setProperty('voice', voice.id)
                            break
        except Exception as e:
            print(f"Error updating TTS settings: {str(e)}")
            
    def stream_text_to_speech(self, text: str, chunk_size: int = 1024) -> Optional[bytes]:
        """
        Stream TTS audio data in chunks.
        
        Args:
            text: Text to convert to speech
            chunk_size: Size of audio chunks in bytes (default: 1024)
            
        Yields:
            bytes: Audio data chunks
            
        Note:
            This is a generator function that yields audio data chunks.
            The last chunk will be None to signal the end of the stream.
        """
        try:
            self._lazy_init_engine()
            buffer = io.BytesIO()
            
            # Configure the wave file
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(self._channels)
                wav_file.setsampwidth(self._sample_width)
                wav_file.setframerate(self._sample_rate)
                
                def callback(data):
                    self._audio_queue.put(data)
                
                # Set up the callback for audio data
                self._engine.connect('data', callback)
                
                # Generate speech
                with self._engine_lock:
                    self._engine.say(text)
                    self._engine.runAndWait()
                
                # Signal end of stream
                self._audio_queue.put(None)
                
                # Read and yield chunks from the queue
                while True:
                    chunk = self._audio_queue.get()
                    if chunk is None:
                        break
                    yield chunk
                    
        except Exception as e:
            yield str(e).encode()
            return
