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

    def test_code_analysis(self, code: str, language: str = None, analysis_type: str = "full"):
        """Test code analysis endpoint with different languages"""
        try:
            print(f"\nTesting code analysis for {language if language else 'auto-detected'} code...")
            
            # Prepare request data
            data = {
                "code": code,
                "analysis_type": analysis_type
            }
            if language:
                data["language"] = language
                
            # Send request
            response = requests.post(
                f"{self.base_url}/code/analyze",
                json=data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("\nAnalysis Results:")
                print(f"Language: {result['language']}")
                print(f"Complexity Score: {result.get('complexity_score', 'N/A')}")
                
                if result.get('suggestions'):
                    print("\nSuggestions:")
                    for suggestion in result['suggestions']:
                        print(f"- {suggestion}")
                        
                if result.get('best_practices'):
                    print("\nBest Practices:")
                    for practice in result['best_practices']:
                        print(f"- {practice}")
                        
                if result.get('security_issues'):
                    print("\nSecurity Issues:")
                    for issue in result['security_issues']:
                        print(f"- {issue}")
                        
                print("\nFull Analysis:")
                print(json.dumps(result['analysis'], indent=2))
            else:
                print(f"\nError: {response.text}\n")
                
        except Exception as e:
            print(f"\nError: {str(e)}\n")

def main():
    """Main function"""
    print("\nJarvis Voice API Test Client")
    print("=" * 50 + "\n")
    
    client = TestClient()
    
    while True:
        print("\nTest Options:")
        print("1. Test Voice Recognition")
        print("2. Test Text Processing")
        print("3. Test Code Analysis")
        print("4. Test WebSocket")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == "1":
            client.test_voice_endpoint()
        elif choice == "2":
            text = input("Enter text to process: ")
            client.test_text_endpoint(text)
        elif choice == "3":
            print("\nSelect a test case:")
            print("1. Python binary search")
            print("2. JavaScript async function")
            print("3. Rust struct implementation")
            print("4. Go HTTP server")
            print("5. Custom code")
            
            test_choice = input("\nEnter test case (1-5): ")
            
            if test_choice == "1":
                code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
                client.test_code_analysis(code, "python")
                
            elif test_choice == "2":
                code = """
async function fetchData() {
    try {
        const response = await fetch('https://api.example.com/data');
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
    }
}
"""
                client.test_code_analysis(code, "javascript")
                
            elif test_choice == "3":
                code = """
struct User {
    username: String,
    email: String,
    active: bool,
}

impl User {
    fn new(username: String, email: String) -> User {
        User {
            username,
            email,
            active: true,
        }
    }
}
"""
                client.test_code_analysis(code, "rust")
                
            elif test_choice == "4":
                code = """
package main

import (
    "fmt"
    "net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, World!")
}

func main() {
    http.HandleFunc("/", handler)
    http.ListenAndServe(":8080", nil)
}
"""
                client.test_code_analysis(code, "go")
                
            elif test_choice == "5":
                code = input("Enter your code: ")
                language = input("Enter language (or press Enter for auto-detection): ")
                analysis_type = input("Enter analysis type (full/syntax/suggestions/security) [default: full]: ")
                
                if not language.strip():
                    language = None
                if not analysis_type.strip():
                    analysis_type = "full"
                    
                client.test_code_analysis(code, language, analysis_type)
                
        elif choice == "4":
            client.test_websocket_endpoint()
        elif choice == "5":
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()