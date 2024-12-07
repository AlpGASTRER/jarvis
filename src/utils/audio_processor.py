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
        
    def process_audio(self, audio_data: sr.AudioData, sample_width: int = 2) -> sr.AudioData:
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
            sr.AudioData: Processed audio data
            
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
            
            # Create new AudioData object with processed audio
            return sr.AudioData(
                processed_data,
                audio_data.sample_rate,
                audio_data.sample_width
            )
            
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            # Return original audio data if processing fails
            return audio_data
