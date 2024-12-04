import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv
import audioop

def test_voice_recognition():
    # Load environment variables
    load_dotenv()
    wit_ai_key = os.getenv('WIT_AI_KEY')
    
    # Initialize the recognizer and TTS engine
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    
    # Minimal processing settings
    recognizer.energy_threshold = 1200  # Higher threshold for cleaner input
    recognizer.dynamic_energy_threshold = False  # Fixed threshold
    recognizer.pause_threshold = 0.8  # Wait longer for clear phrases
    recognizer.phrase_threshold = 0.5  # More strict phrase detection
    recognizer.non_speaking_duration = 0.5  # Longer silence detection
    
    # Configure TTS
    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1.0)

    def enhance_audio(audio_data):
        """Minimal audio processing for maximum clarity"""
        try:
            # Only convert to mono if needed
            if len(audio_data.frame_data) > audio_data.sample_width * audio_data.sample_rate * 5:
                audio_data.frame_data = audioop.tomono(audio_data.frame_data, audio_data.sample_width, 1, 1)
            
            # Simple DC offset removal
            audio_data.frame_data = audioop.bias(audio_data.frame_data, audio_data.sample_width, 0)
            
            return audio_data
        except:
            return audio_data

    def speak(text):
        print(f"Assistant: {text}")
        engine.say(text)
        engine.runAndWait()
    
    # Main recognition loop
    speak("Ready for voice input. Say 'exit' to stop.")
    
    with sr.Microphone() as source:
        print("\nCalibrating...")
        # Longer initial calibration
        recognizer.adjust_for_ambient_noise(source, duration=3)
        print("Ready!")
        
        while True:
            try:
                print("\nListening...")
                # Longer timeout for better phrase capture
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=None)
                
                try:
                    text = recognizer.recognize_wit(enhance_audio(audio), key=wit_ai_key, show_all=False)
                    print(f"✓ '{text}'")
                    
                    if text.lower().strip() == 'exit':
                        speak("Goodbye!")
                        break
                    
                    speak(f"I heard: {text}")
                    
                except sr.UnknownValueError:
                    print("×")
                except sr.RequestError:
                    print("!")
                    
            except sr.WaitTimeoutError:
                continue
            except KeyboardInterrupt:
                speak("Stopping.")
                break
            except:
                continue

if __name__ == "__main__":
    print("Starting voice recognition...")
    print("Press Ctrl+C to exit")
    test_voice_recognition()
