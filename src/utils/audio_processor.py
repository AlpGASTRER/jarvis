import numpy as np
import noisereduce as nr
from webrtcvad import Vad
import audioop
from typing import Any
import speech_recognition as sr

class AudioProcessor:
    def __init__(self):
        self.vad = Vad(2)  # Moderate aggressiveness
        self.sample_rate = 16000
        
    def process_audio(self, audio_data: sr.AudioData, sample_width: int = 2) -> sr.AudioData:
        """
        Light audio processing pipeline that preserves original quality
        """
        try:
            # Get raw audio data
            raw_data = audio_data.get_raw_data()
            
            # Convert to numpy array
            audio_array = np.frombuffer(raw_data, dtype=np.int16)
            
            # Convert to mono if stereo
            if audio_data.sample_width * audio_data.sample_rate < len(raw_data):
                raw_data = audioop.tomono(raw_data, audio_data.sample_width, 1, 1)
                audio_array = np.frombuffer(raw_data, dtype=np.int16)
            
            # Remove DC offset
            raw_data = audioop.bias(raw_data, audio_data.sample_width, 0)
            
            # Only apply noise reduction if significant noise detected
            if np.std(audio_array) > 500:
                # Convert to float32 for noise reduction
                float_audio = audio_array.astype(np.float32) / 32768.0
                
                # Apply gentle noise reduction
                reduced_noise = nr.reduce_noise(
                    y=float_audio,
                    sr=self.sample_rate,
                    prop_decrease=0.3,
                    stationary=True,
                    n_jobs=2
                )
                
                # Convert back to int16
                processed_array = np.int16(reduced_noise * 32768.0)
                processed_data = processed_array.tobytes()
            else:
                processed_data = raw_data
            
            # Create new AudioData object with processed audio
            return sr.AudioData(
                processed_data,
                audio_data.sample_rate,
                audio_data.sample_width
            )
            
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            return audio_data
