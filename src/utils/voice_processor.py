"""
Voice Processing Module

This module handles voice recognition and audio processing tasks for the Jarvis AI Assistant.
It provides functionality for processing audio data, noise reduction, and speech recognition
using the Wit.ai service.

Key Features:
- Audio format conversion
- Noise reduction using noisereduce
- Speech recognition via Wit.ai
- Support for base64 encoded audio
- Error handling and fallback strategies

Dependencies:
- speech_recognition: For speech recognition
- numpy: For audio data manipulation
- noisereduce: For noise reduction
- wave: For WAV file handling
"""

import speech_recognition as sr
import numpy as np
import noisereduce as nr
from typing import Union, Tuple
import io
import wave
import base64
import os

class VoiceProcessor:
    """
    A class for processing voice input and performing speech recognition.
    
    This class handles audio data conversion, enhancement, and speech recognition
    using the Wit.ai service. It supports both raw and base64 encoded audio data,
    and includes noise reduction capabilities for improved recognition accuracy.
    
    Attributes:
        recognizer: SpeechRecognition recognizer instance
        model: Gemini AI model instance
        tts_engine: pyttsx3 TTS engine instance
    """
    
    def __init__(self):
        """Initialize the voice processor with a speech recognizer and AI model."""
        self.recognizer = sr.Recognizer()
        
        # Initialize Gemini AI once
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize TTS engine
        import pyttsx3
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 175)  # Speed of speech
        self.tts_engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

        # Pre-warm the model with a dummy request
        try:
            self.model.generate_content("Hello")
        except Exception as e:
            print(f"Model pre-warming failed: {e}")
        
    def _convert_to_audio_data(self, audio_bytes: bytes, sample_rate: int, channels: int) -> sr.AudioData:
        """
        Convert raw audio bytes to SpeechRecognition AudioData format.
        
        Args:
            audio_bytes: Raw audio data in bytes
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            
        Returns:
            sr.AudioData: Audio data in SpeechRecognition format
            
        Note:
            Uses 16-bit PCM format for audio data
        """
        # Create an in-memory wave file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)  # 16-bit audio
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
            
        # Read the wave file into an AudioData object
        wav_buffer.seek(0)
        with wave.open(wav_buffer, 'rb') as wav_file:
            audio_data = sr.AudioData(
                wav_file.readframes(wav_file.getnframes()),
                wav_file.getframerate(),
                wav_file.getsampwidth()
            )
        return audio_data

    def process_base64_audio(self, audio_base64: str, sample_rate: int = 16000, channels: int = 1) -> Tuple[sr.AudioData, sr.AudioData]:
        """
        Process base64 encoded audio data and apply noise reduction.
        
        Args:
            audio_base64: Base64 encoded audio data
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            
        Returns:
            Tuple containing:
                - Original audio data (sr.AudioData)
                - Enhanced audio data with noise reduction (sr.AudioData)
        """
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Convert to AudioData format
        original_audio = self._convert_to_audio_data(audio_bytes, sample_rate, channels)
        
        # Convert to numpy array for noise reduction processing
        audio_array = np.frombuffer(original_audio.frame_data, dtype=np.int16)
        
        # Apply noise reduction using noisereduce library
        reduced_noise = nr.reduce_noise(
            y=audio_array.astype(float),
            sr=sample_rate,
            stationary=True,  # Assume stationary noise
            prop_decrease=0.75  # Noise reduction strength
        )
        
        # Convert back to 16-bit PCM format
        enhanced_audio_bytes = (reduced_noise * 32767).astype(np.int16).tobytes()
        
        # Create enhanced AudioData object
        enhanced_audio = self._convert_to_audio_data(enhanced_audio_bytes, sample_rate, channels)
        
        return original_audio, enhanced_audio
        
    def recognize_wit(self, audio_data: sr.AudioData, wit_key: str) -> str:
        """
        Perform speech recognition using Wit.ai service.
        
        Args:
            audio_data: Audio data in SpeechRecognition format
            wit_key: Wit.ai API key
            
        Returns:
            str: Recognized text
            
        Raises:
            ValueError: If speech couldn't be understood
            RuntimeError: If Wit.ai service request fails
        """
        try:
            return self.recognizer.recognize_wit(audio_data, key=wit_key)
        except sr.UnknownValueError:
            raise ValueError("Could not understand audio")
        except sr.RequestError as e:
            raise RuntimeError(f"Could not request results from Wit.ai service; {str(e)}")
            
    def process_voice(self, audio_base64: str, wit_key: str, sample_rate: int = 16000, channels: int = 1) -> dict:
        """
        Process voice data and perform speech recognition with fallback strategy.
        
        This method attempts recognition on original audio first, then falls back
        to noise-reduced audio if the first attempt fails.
        
        Args:
            audio_base64: Base64 encoded audio data
            wit_key: Wit.ai API key
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            
        Returns:
            dict: Recognition results containing:
                - success: Boolean indicating success
                - text: Recognized text (if successful)
                - audio_used: Which audio was used ("original" or "enhanced")
                - error: Error message (if failed)
        """
        try:
            # Process audio and apply noise reduction
            original_audio, enhanced_audio = self.process_base64_audio(audio_base64, sample_rate, channels)
            
            # Try recognition with original audio first
            try:
                text = self.recognize_wit(original_audio, wit_key)
                audio_used = "original"
            except (ValueError, RuntimeError):
                # Fallback to enhanced audio if original fails
                text = self.recognize_wit(enhanced_audio, wit_key)
                audio_used = "enhanced"
                
            return {
                "success": True,
                "text": text,
                "audio_used": audio_used
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def recognize_speech(self, audio_data: bytes) -> str:
        """
        Recognize speech from audio data using Wit.ai.
        
        Args:
            audio_data: Raw audio data in bytes
            
        Returns:
            str: Recognized text, or None if recognition failed
        """
        try:
            # Get Wit.ai API key
            wit_key = os.getenv('WIT_EN_KEY')
            if not wit_key:
                print("Wit.ai API key not found")
                return None
                
            # Convert audio data to AudioData format
            audio = self._convert_to_audio_data(audio_data, 16000, 1)
            
            # Use Wit.ai to recognize speech
            text = self.recognizer.recognize_wit(
                audio,
                key=wit_key
            )
            return text
            
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Wit.ai; {e}")
            return None
        except Exception as e:
            print(f"Error recognizing speech: {e}")
            return None

    def get_ai_response(self, text: str) -> str:
        """
        Get AI response for the recognized text.
        
        Args:
            text: Recognized text to process
            
        Returns:
            str: AI response
        """
        try:
            # Get response using pre-initialized model
            response = self.model.generate_content(text)
            return response.text
            
        except Exception as e:
            print(f"Error getting AI response: {e}")
            return "I apologize, but I couldn't process your request at the moment."

    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using pyttsx3.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            bytes: WAV audio data, or None if conversion fails
        """
        try:
            import io
            import wave
            
            # Create an in-memory buffer
            buffer = io.BytesIO()
            
            # Save speech to the buffer
            self.tts_engine.save_to_file(text, 'temp.wav')
            self.tts_engine.runAndWait()
            
            # Read the generated file
            with open('temp.wav', 'rb') as f:
                audio_data = f.read()
            
            # Clean up
            import os
            os.remove('temp.wav')
            
            return audio_data
            
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return None

    def stream_tts(self, text: str, chunk_size: int = 4096):
        """
        Stream text-to-speech conversion in chunks.
        
        Args:
            text: Text to convert to speech
            chunk_size: Size of each audio chunk in bytes
            
        Yields:
            bytes: Chunks of MP3 audio data
        """
        try:
            import io
            import wave
            
            # Create an in-memory buffer
            buffer = io.BytesIO()
            
            # Save speech to the buffer
            self.tts_engine.save_to_file(text, 'temp.wav')
            self.tts_engine.runAndWait()
            
            # Read the generated file
            with open('temp.wav', 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            
            # Clean up
            import os
            os.remove('temp.wav')
                
        except Exception as e:
            print(f"Error in TTS streaming: {e}")
            return None
