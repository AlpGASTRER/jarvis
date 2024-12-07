import speech_recognition as sr
import numpy as np
import noisereduce as nr
from typing import Union, Tuple
import io
import wave
import base64

class VoiceProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        
    def _convert_to_audio_data(self, audio_bytes: bytes, sample_rate: int, channels: int) -> sr.AudioData:
        """Convert raw audio bytes to AudioData object"""
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
        """Process base64 encoded audio and return both original and enhanced AudioData"""
        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)
        
        # Convert to AudioData
        original_audio = self._convert_to_audio_data(audio_bytes, sample_rate, channels)
        
        # Convert to numpy array for noise reduction
        audio_array = np.frombuffer(original_audio.frame_data, dtype=np.int16)
        
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(
            y=audio_array.astype(float),
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.75
        )
        
        # Convert back to 16-bit PCM
        enhanced_audio_bytes = (reduced_noise * 32767).astype(np.int16).tobytes()
        
        # Create enhanced AudioData
        enhanced_audio = self._convert_to_audio_data(enhanced_audio_bytes, sample_rate, channels)
        
        return original_audio, enhanced_audio
        
    def recognize_wit(self, audio_data: sr.AudioData, wit_key: str) -> str:
        """Recognize speech using Wit.ai"""
        try:
            return self.recognizer.recognize_wit(audio_data, key=wit_key)
        except sr.UnknownValueError:
            raise ValueError("Could not understand audio")
        except sr.RequestError as e:
            raise RuntimeError(f"Could not request results from Wit.ai service; {str(e)}")
            
    def process_voice(self, audio_base64: str, wit_key: str, sample_rate: int = 16000, channels: int = 1) -> dict:
        """Process voice data and return recognition results"""
        try:
            # Process audio
            original_audio, enhanced_audio = self.process_base64_audio(audio_base64, sample_rate, channels)
            
            # Try recognition with original audio first
            try:
                text = self.recognize_wit(original_audio, wit_key)
                audio_used = "original"
            except (ValueError, RuntimeError):
                # Fallback to enhanced audio
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
