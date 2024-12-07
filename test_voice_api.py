import pyaudio
import wave
import base64
import requests
import json
import websockets
import asyncio
import keyboard
import time
from colorama import init, Fore, Style
import pygame
import speech_recognition as sr
import os

# Initialize colorama
init()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class AudioPlayer:
    def __init__(self):
        pass

    def play_audio(self, audio_data):
        """Play audio from bytes data"""
        try:
            # Save audio data to a temporary file with proper WAV headers
            temp_file = "temp_audio.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(22050)  # Sample rate
                wf.writeframes(audio_data)
            
            # Play the audio
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            # Clean up
            os.remove(temp_file)
            
        except Exception as e:
            print(f"{Fore.RED}Error playing audio: {str(e)}{Style.RESET_ALL}")

class TestClient:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        pygame.mixer.init()
        
    def record_audio(self):
        """Record audio from microphone and return the audio data"""
        print("\n* Recording...")
        print("Press and hold SPACE to record, release to stop")
        
        r = sr.Recognizer()
        with sr.Microphone(sample_rate=RATE) as source:
            # Wait for SPACE key press
            keyboard.wait('space', suppress=True)
            print("Recording started...")
            
            # Record until SPACE is released
            audio = r.listen(source)
            print("Recording stopped\n")
            
            # Get raw audio data
            return audio.get_wav_data()

    def test_voice_endpoint(self):
        """Test POST /voice endpoint"""
        try:
            print("Testing POST /voice endpoint...")
            
            # Record audio
            audio_data = self.record_audio()
            if not audio_data:
                return
            
            # Create form data with WAV audio
            files = {'audio_file': ('audio.wav', audio_data, 'audio/wav')}
            data = {'enhance_audio': 'false'}
            
            # Send request
            response = requests.post(f"{self.base_url}/voice", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                print("\nResponse:")
                print(f"Recognized Text: {result.get('recognized_text', 'N/A')}")
                print(f"Audio Used: {result.get('audio_used', 'N/A')}")
                print(f"AI Response: {result.get('ai_response', 'N/A')}\n")
            else:
                print(f"\nError: {response.text}\n")
                
        except Exception as e:
            print(f"\nError: {str(e)}\n")

    def test_websocket_endpoint(self):
        """Test the /ws/voice WebSocket endpoint"""
        print("Testing WebSocket endpoint...")
        
        try:
            # Record audio
            audio_data = self.record_audio()
            if not audio_data:
                return
                
            async def test_ws():
                uri = f"{self.base_url.replace('http', 'ws')}/ws/voice"
                async with websockets.connect(uri) as websocket:
                    # Send raw audio bytes
                    await websocket.send(audio_data)
                    print("Sent audio data, waiting for response...")
                    
                    # Get response
                    response = await websocket.recv()
                    result = json.loads(response)
                    
                    print("\nResponse:")
                    print(f"Text: {result.get('text', 'N/A')}")
                    print(f"Response: {result.get('response', 'N/A')}\n")
                    
            asyncio.get_event_loop().run_until_complete(test_ws())
            
        except Exception as e:
            print(f"\nError: {str(e)}\n")

    def test_tts_endpoint(self, text="Hello, I am Jarvis. How can I help you today?"):
        """Test the /tts endpoint"""
        print("Testing TTS endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/tts", params={"text": text}, stream=True)
            if response.status_code == 200:
                print("Successfully received audio response")
                
                # Save to a unique temporary file
                temp_file = f"temp_audio_{int(time.time()*1000)}.wav"
                with open(temp_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                try:
                    # Play audio using pygame
                    pygame.mixer.music.load(temp_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)
                finally:
                    # Clean up
                    pygame.mixer.music.unload()
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    """Main function"""
    print("\nJarvis Voice API Test Client")
    print("=" * 50 + "\n")
    
    client = TestClient()
    
    while True:
        print("Options:")
        print("1. Test POST /voice endpoint")
        print("2. Test WebSocket endpoint")
        print("3. Test TTS endpoint")
        print("4. Exit\n")
        
        choice = input("Enter your choice (1-4): ")
        if choice == "4":
            break
            
        if choice == "1":
            client.test_voice_endpoint()
        elif choice == "2":
            client.test_websocket_endpoint()
        elif choice == "3":
            client.test_tts_endpoint()

if __name__ == "__main__":
    main()
