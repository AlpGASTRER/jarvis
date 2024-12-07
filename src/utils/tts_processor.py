import pyttsx3
import io
import wave
import base64
import threading
from typing import Optional
import queue
import time

class TTSProcessor:
    def __init__(self):
        self._engine = None
        self._engine_lock = threading.Lock()
        self._audio_queue = queue.Queue()
        self._sample_width = 2  # 16-bit audio
        self._sample_rate = 22050  # Standard TTS sample rate
        self._channels = 1  # Mono audio
        
    def _lazy_init_engine(self):
        """Initialize the TTS engine if not already initialized"""
        if self._engine is None:
            with self._engine_lock:
                if self._engine is None:  # Double-check pattern
                    self._engine = pyttsx3.init()
                    self._engine.setProperty('rate', 175)
                    self._engine.setProperty('volume', 1.0)
                    
                    # Configure voice (neutral voice)
                    voices = self._engine.getProperty('voices')
                    for voice in voices:
                        if 'david' in voice.name.lower() or 'mark' in voice.name.lower():
                            self._engine.setProperty('voice', voice.id)
                            break
    
    def text_to_speech_base64(self, text: str, retry_count: int = 3) -> dict:
        """Convert text to speech and return as base64 encoded audio"""
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
                
                # Set up the callback
                self._engine.connect('data', callback)
                
                # Generate speech
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
                        
            # Get the audio data and convert to base64
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
        """Convert text to speech and return raw audio bytes"""
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
                
                # Set up the callback
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
        """Update TTS engine settings"""
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
        """Stream TTS audio data in chunks"""
        try:
            self._lazy_init_engine()
            buffer = io.BytesIO()
            
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(self._channels)
                wav_file.setsampwidth(self._sample_width)
                wav_file.setframerate(self._sample_rate)
                
                def callback(data):
                    self._audio_queue.put(data)
                
                self._engine.connect('data', callback)
                
                with self._engine_lock:
                    self._engine.say(text)
                    self._engine.runAndWait()
                
                # Signal end of stream
                self._audio_queue.put(None)
                
                # Read chunks from the queue
                while True:
                    chunk = self._audio_queue.get()
                    if chunk is None:
                        break
                    yield chunk
                    
        except Exception as e:
            yield str(e).encode()
            return
