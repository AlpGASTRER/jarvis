"""
Audio Processing Module

This module provides audio processing capabilities for the Jarvis AI Assistant.
It handles various audio enhancement tasks including noise reduction, DC offset
removal, and mono conversion while preserving audio quality.

Key Features:
- Adaptive noise reduction
- DC offset removal
- Stereo to mono conversion
- Voice activity detection
- Quality-preserving processing

Dependencies:
- numpy: For audio data manipulation
- noisereduce: For noise reduction
- webrtcvad: For voice activity detection
- audioop: For audio operations
"""

import numpy as np
import noisereduce as nr
from webrtcvad import Vad
import audioop
from typing import Any
import speech_recognition as sr
import io
import wave

class AudioProcessor:
    """
    A class for processing and enhancing audio data.
    
    This class provides methods for improving audio quality while preserving
    important speech characteristics. It uses WebRTC's Voice Activity Detection
    and applies adaptive noise reduction based on signal characteristics.
    
    Attributes:
        vad: WebRTC Voice Activity Detector
        sample_rate: Audio sample rate in Hz
    """
    
    def __init__(self):
        """
        Initialize the audio processor.
        
        Sets up the Voice Activity Detector with moderate aggressiveness
        and configures the default sample rate for processing.
        """
        self.vad = Vad(2)  # Moderate aggressiveness (0=least, 3=most)
        self.sample_rate = 16000  # 16kHz sample rate for optimal speech recognition
        
    def validate_for_gemini(self, audio_data: np.ndarray, sample_rate: int) -> None:
        """
        Validate audio parameters for Gemini compatibility
        """
        if sample_rate not in [16000, 24000, 48000]:
            raise ValueError(f"Unsupported sample rate {sample_rate}Hz. Gemini requires 16k, 24k or 48k")
            
    def convert_to_gemini_format(self, audio_data: np.ndarray, sample_rate: int) -> bytes:
        """
        Convert processed audio to Gemini-compatible WAV format
        """
        with io.BytesIO() as wav_buffer:
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
            return wav_buffer.getvalue()
            
    def process_audio(self, audio_data: sr.AudioData, sample_width: int = 2) -> bytes:
        """
        Process audio data with quality-preserving enhancements.
        
        This method applies a series of audio processing steps:
        1. Converts stereo to mono if needed
        2. Removes DC offset
        3. Applies adaptive noise reduction based on signal characteristics
        
        Args:
            audio_data: Input audio data in SpeechRecognition format
            sample_width: Audio sample width in bytes (default: 2 for 16-bit)
            
        Returns:
            bytes: Processed audio data in Gemini-compatible WAV format
            
        Note:
            The processing pipeline is designed to be light and preserve
            original audio quality while improving speech clarity.
        """
        try:
            # Extract raw audio data from AudioData object
            raw_data = audio_data.get_raw_data()
            
            # Convert raw bytes to numpy array for processing
            audio_array = np.frombuffer(raw_data, dtype=np.int16)
            
            # Convert stereo to mono if needed
            # Check if audio is stereo by comparing data length with expected mono length
            if audio_data.sample_width * audio_data.sample_rate < len(raw_data):
                raw_data = audioop.tomono(raw_data, audio_data.sample_width, 1, 1)
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
            
            # Remove DC offset to center the waveform around zero
            raw_data = audioop.bias(raw_data, audio_data.sample_width, 0)
            
            # Apply noise reduction only if significant noise is detected
            # Use standard deviation as a measure of signal variation
            if np.std(audio_array) > 500:  # Threshold for noise detection
                # Convert to float32 [-1.0, 1.0] range for noise reduction
                float_audio = audio_array.astype(np.float32) / 32768.0
                
                # Apply gentle noise reduction to preserve speech quality
                reduced_noise = nr.reduce_noise(
                    y=float_audio,
                    sr=self.sample_rate,
                    prop_decrease=0.3,  # Conservative noise reduction
                    stationary=True,    # Assume relatively constant noise
                    n_jobs=2            # Parallel processing
                )
                
                # Convert back to 16-bit PCM format
                processed_array = np.int16(reduced_noise * 32768.0)
                processed_data = processed_array.tobytes()
            else:
                # If noise level is low, use the DC-offset corrected data
                processed_data = raw_data
            
            # Validate audio parameters for Gemini compatibility
            self.validate_for_gemini(np.frombuffer(processed_data, dtype=np.int16), audio_data.sample_rate)
            
            # Convert processed audio to Gemini-compatible WAV format
            return self.convert_to_gemini_format(np.frombuffer(processed_data, dtype=np.int16), audio_data.sample_rate)
            
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            # Return original audio data if processing fails
            return audio_data.get_raw_data()
