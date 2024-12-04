import speech_recognition as sr
import pyaudio
import wave
import time

def test_microphone():
    print("Testing microphone setup...")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Show all audio devices
    print("\nAvailable Audio Devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"\nDevice {i}:")
        print(f"  Name: {info['name']}")
        print(f"  Input Channels: {info['maxInputChannels']}")
        print(f"  Output Channels: {info['maxOutputChannels']}")
        print(f"  Default Sample Rate: {info['defaultSampleRate']}")
    
    # Get default input device
    try:
        default_input = p.get_default_input_device_info()
        print(f"\nDefault Input Device: {default_input['name']}")
    except Exception as e:
        print(f"\nError getting default input device: {e}")
    
    # Test recording
    print("\nTesting recording capability...")
    try:
        # Record settings
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 3
        
        # Start recording
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        print("\nRecording for 3 seconds...")
        frames = []
        
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("Recording finished!")
        
        # Stop recording
        stream.stop_stream()
        stream.close()
        
        # Save the recorded data as a WAV file
        WAVE_OUTPUT_FILENAME = "test_recording.wav"
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"\nRecording saved as {WAVE_OUTPUT_FILENAME}")
        
    except Exception as e:
        print(f"Error during recording test: {e}")
    
    # Test speech recognition
    print("\nTesting speech recognition...")
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("\nAdjusting for ambient noise... Please wait...")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            print("\nPlease speak a simple phrase like 'Hello World'...")
            audio = recognizer.listen(source, timeout=5)
            print("\nProcessing speech...")
            
            try:
                text = recognizer.recognize_google(audio)
                print(f"Recognized text: {text}")
                print("\nSpeech recognition test successful!")
            except sr.UnknownValueError:
                print("Could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results from speech recognition service; {e}")
    
    except Exception as e:
        print(f"Error during speech recognition test: {e}")
    
    finally:
        p.terminate()

if __name__ == "__main__":
    test_microphone()
