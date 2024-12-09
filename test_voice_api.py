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
import io

# Initialize colorama
init()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()

    def play_audio(self, audio_data):
        """Play audio from bytes data"""
        try:
            # Save audio data to a temporary file
            temp_wav = "temp_audio.wav"
            with open(temp_wav, 'wb') as f:
                f.write(audio_data)
            
            try:
                # Play the audio
                pygame.mixer.music.load(temp_wav)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            finally:
                # Clean up
                pygame.mixer.music.unload()
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            
        except Exception as e:
            print(f"{Fore.RED}Error playing audio: {str(e)}{Style.RESET_ALL}")

class TestClient:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        
    def record_audio(self):
        """Record audio from microphone and return the audio data"""
        print("\n* Recording...")
        print("Press and hold SPACE to record, release to stop")
        
        try:
            r = sr.Recognizer()
            with sr.Microphone(sample_rate=RATE) as source:
                # Adjust for ambient noise
                print("Adjusting for ambient noise...")
                r.adjust_for_ambient_noise(source, duration=1)
                
                # Wait for SPACE key press
                print("Ready! Press and hold SPACE to record")
                keyboard.wait('space', suppress=True)
                print("Recording started...")
                
                # Start recording
                audio = None
                try:
                    # Record until SPACE is released
                    while keyboard.is_pressed('space'):
                        audio = r.listen(source, timeout=1, phrase_time_limit=None)
                        if audio:
                            break
                    
                    print("Recording stopped\n")
                    
                    # Get raw audio data if we have it
                    if audio:
                        return audio.get_wav_data()
                    else:
                        print("No audio data captured")
                        return None
                        
                except sr.WaitTimeoutError:
                    print("Recording timed out")
                    return None
                except Exception as e:
                    print(f"Error during recording: {e}")
                    return None
                    
        except Exception as e:
            print(f"Error initializing audio: {e}")
            return None

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

    def test_text_endpoint(self, text: str):
        """Test POST /text endpoint"""
        try:
            print("\nTesting POST /text endpoint...")
            
            # Prepare request data
            data = {
                "text": text,
                "mode": "general",
                "return_audio": False
            }
            
            # Send request
            response = requests.post(f"{self.base_url}/text", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print("\nResponse:")
                print(f"AI Response: {result.get('response', 'N/A')}")
                
                # If audio response is present
                if result.get('audio_response'):
                    print("\nPlaying audio response...")
                    audio_data = base64.b64decode(result['audio_response']['audio_base64'])
                    player = AudioPlayer()
                    player.play_audio(audio_data)
            else:
                print(f"\nError: {response.text}\n")
                
        except Exception as e:
            print(f"\nError: {str(e)}\n")

    def test_conversation_endpoint(self, action: str, text: str = None, chat_id: str = None):
        """Test POST /conversation endpoint"""
        try:
            print("\nTesting POST /conversation endpoint...")
            
            # Prepare request data
            data = {
                "action": action,
                "text": text,
                "chat_id": chat_id
            }
            
            # Send request
            response = requests.post(f"{self.base_url}/conversation", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print("\nResponse:")
                if result.get('chat_id'):
                    print(f"Chat ID: {result['chat_id']}")
                if result.get('response'):
                    print(f"AI Response: {result['response']}")
                if result.get('message'):
                    print(f"Message: {result['message']}")
                    
                if action == "list":
                    print("\nActive Chats:")
                    for chat in result.get('chats', []):
                        status = "(Active)" if chat['is_active'] else ""
                        print(f"\nChat {chat['id']} {status}")
                        print("History:")
                        for msg in chat['history']:
                            role = msg['role']
                            text = msg['text']
                            print(f"{Fore.GREEN if role == 'assistant' else Fore.BLUE}{role}: {text}{Style.RESET_ALL}")
                else:
                    print("\nConversation History:")
                    for msg in result.get('history', []):
                        role = msg['role']
                        text = msg['text']
                        print(f"{Fore.GREEN if role == 'assistant' else Fore.BLUE}{role}: {text}{Style.RESET_ALL}")
            else:
                print(f"\nError: {response.text}\n")
                
            return response.json()
        except Exception as e:
            print(f"\nError: {str(e)}\n")

    async def test_websocket_conversation(self, websocket):
        """Handle conversation in WebSocket connection"""
        try:
            # Get recognition result
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("type") == "recognition":
                print("\nRecognition Result:")
                print(f"Success: {result.get('success', False)}")
                print(f"Text: {result.get('text', 'N/A')}")
                
                # Get AI response if recognition was successful
                if result.get("success"):
                    response = await websocket.recv()
                    result = json.loads(response)
                    
                    if result.get("type") == "response":
                        print("\nAI Response:")
                        print(f"Text: {result.get('text', 'N/A')}")
                        print("\nConversation History:")
                        for msg in result.get('history', []):
                            role = msg['role']
                            text = msg['text']
                            print(f"{Fore.GREEN if role == 'assistant' else Fore.BLUE}{role}: {text}{Style.RESET_ALL}")
                        
                        # If audio response is present
                        if result.get("audio"):
                            print("\nPlaying audio response...")
                            audio_data = base64.b64decode(result["audio"])
                            player = AudioPlayer()
                            player.play_audio(audio_data)
                            
        except Exception as e:
            print(f"Error in WebSocket conversation: {e}")

    def test_websocket_endpoint(self):
        """Test the /ws/voice WebSocket endpoint"""
        print("Testing WebSocket endpoint...")
        
        # Record audio first
        audio_data = self.record_audio()
        if audio_data is None:
            print("Failed to record audio")
            return
            
        async def test_ws():
            uri = f"{self.base_url.replace('http', 'ws')}/ws/voice"
            async with websockets.connect(uri) as websocket:
                # Send audio data as JSON with proper format
                message = {
                    "type": "audio",
                    "audio": base64.b64encode(audio_data).decode(),
                    "return_audio": True
                }
                await websocket.send(json.dumps(message))
                print("Sent audio data, waiting for response...")
                
                await self.test_websocket_conversation(websocket)
                
        asyncio.get_event_loop().run_until_complete(test_ws())

    def test_tts_endpoint(self, text="Hello, I am Jarvis. How can I help you today?"):
        """Test the /tts endpoint"""
        print("Testing TTS endpoint...")
        
        try:
            response = requests.get(f"{self.base_url}/tts", params={"text": text}, stream=True)
            if response.status_code == 200:
                print("Successfully received audio response")
                
                # Save to a unique temporary file
                temp_file = f"temp_audio_{int(time.time()*1000)}.mp3"
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
    """Main function to run tests"""
    client = TestClient()
    current_chat_id = None
    
    while True:
        print("\n=== Jarvis API Test Menu ===")
        print("1. Test Voice Recognition")
        print("2. Test Text Processing")
        print("3. Test Code Analysis")
        print("4. Test WebSocket")
        print("5. Test Conversation")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            client.test_voice_endpoint()
        elif choice == "2":
            text = input("\nEnter text to process: ")
            client.test_text_endpoint(text)
        elif choice == "3":
            code = input("\nEnter code or question: ")
            client.test_text_endpoint(code)
        elif choice == "4":
            client.test_websocket_endpoint()
        elif choice == "5":
            while True:
                print("\nConversation Test Menu:")
                if current_chat_id:
                    print(f"Current Chat: {current_chat_id}")
                print("1. Start New Conversation")
                print("2. Continue Conversation")
                print("3. Clear History")
                print("4. List Active Chats")
                print("5. Switch Chat")
                print("6. Back to Main Menu")
                
                sub_choice = input("\nEnter your choice (1-6): ")
                
                if sub_choice == "1":
                    response = client.test_conversation_endpoint("start")
                    if response and response.get('chat_id'):
                        current_chat_id = response.get('chat_id')
                elif sub_choice == "2":
                    text = input("Enter your message: ")
                    client.test_conversation_endpoint("continue", text, current_chat_id)
                elif sub_choice == "3":
                    client.test_conversation_endpoint("clear", chat_id=current_chat_id)
                elif sub_choice == "4":
                    client.test_conversation_endpoint("list")
                elif sub_choice == "5":
                    chat_id = input("Enter chat ID to switch to: ")
                    response = client.test_conversation_endpoint("switch", chat_id=chat_id)
                    if response and response.get('success'):
                        current_chat_id = chat_id
                elif sub_choice == "6":
                    break
                else:
                    print("Invalid choice!")
        elif choice == "6":
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice!")

if __name__ == "__main__":
    main()
