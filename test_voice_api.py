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
import audioop

# Initialize colorama
init()

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

class AudioPlayer:
    def __init__(self):
        """Initialize the audio player with pygame mixer"""
        try:
            # Make sure pygame is initialized
            if not pygame.get_init():
                pygame.init()
            
            # Reset and reinitialize the mixer with specific settings
            pygame.mixer.quit()
            pygame.mixer.init(
                frequency=44100,  # Standard frequency
                size=-16,         # 16-bit signed
                channels=2,       # Stereo
                buffer=2048       # Smaller buffer for less latency
            )
            
            # Set volume to maximum
            pygame.mixer.music.set_volume(1.0)
            print(f"{Fore.GREEN}Audio player initialized successfully{Style.RESET_ALL}")
            print(f"Mixer settings: {pygame.mixer.get_init()}")
            
        except Exception as e:
            print(f"{Fore.RED}Failed to initialize audio player: {str(e)}{Style.RESET_ALL}")
            print(f"Pygame initialization status: {pygame.get_init()}")
            print(f"Mixer initialization status: {pygame.mixer.get_init()}")

    async def play_audio(self, audio_data):
        """Play audio from bytes data asynchronously."""
        temp_files = []  # Keep track of temporary files
        try:
            # Handle base64 input
            if isinstance(audio_data, str):
                print(f"{Fore.CYAN}Decoding base64 audio data...{Style.RESET_ALL}")
                audio_data = base64.b64decode(audio_data)
            
            print(f"{Fore.CYAN}Audio data size: {len(audio_data)} bytes{Style.RESET_ALL}")
            
            # Create unique filenames using absolute paths
            temp_wav = os.path.abspath(os.path.join(os.getcwd(), f"temp_audio_{int(time.time()*1000)}.wav"))
            temp_files.append(temp_wav)
            
            # Save initial audio data
            with open(temp_wav, 'wb') as f:
                f.write(audio_data)
            print(f"{Fore.GREEN}Saved audio to {temp_wav}{Style.RESET_ALL}")
            
            # Read and convert audio if needed
            with wave.open(temp_wav, 'rb') as wav_file:
                params = wav_file.getparams()
                print(f"Input WAV parameters: {params}")
                
                # Check if conversion is needed
                needs_conversion = params.framerate != 44100 or params.nchannels != 2
                
                if needs_conversion:
                    print(f"{Fore.YELLOW}Converting audio format...{Style.RESET_ALL}")
                    
                    # Read all audio data
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    
                    # Convert sample rate if needed
                    if params.framerate != 44100:
                        print(f"Converting sample rate from {params.framerate} to 44100")
                        audio_data, _ = audioop.ratecv(audio_data, params.sampwidth, 
                                                     params.nchannels, params.framerate,
                                                     44100, None)
                    
                    # Convert to stereo if mono
                    if params.nchannels == 1:
                        print("Converting mono to stereo")
                        audio_data = audioop.tostereo(audio_data, params.sampwidth, 1, 1)
                    
                    # Create new temp file for converted audio
                    converted_wav = os.path.abspath(os.path.join(os.getcwd(), f"temp_audio_converted_{int(time.time()*1000)}.wav"))
                    temp_files.append(converted_wav)
                    
                    # Write converted audio
                    with wave.open(converted_wav, 'wb') as out_wav:
                        out_wav.setnchannels(2)
                        out_wav.setsampwidth(params.sampwidth)
                        out_wav.setframerate(44100)
                        out_wav.writeframes(audio_data)
                    
                    print(f"{Fore.GREEN}Audio conversion complete{Style.RESET_ALL}")
                    
                    # Use the converted file for playback
                    playback_file = converted_wav
                else:
                    # Use original file if no conversion needed
                    playback_file = temp_wav
            
            # Play the audio
            print(f"{Fore.CYAN}Loading audio file...{Style.RESET_ALL}")
            pygame.mixer.music.load(playback_file)
            print(f"{Fore.CYAN}Playing audio...{Style.RESET_ALL}")
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            start_time = time.time()
            duration = params.nframes / params.framerate  # Calculate audio duration
            print(f"Audio duration: {duration:.2f} seconds")
            
            while pygame.mixer.music.get_busy() or (time.time() - start_time) < duration:
                await asyncio.sleep(0.1)
            
            print(f"{Fore.GREEN}Audio playback completed{Style.RESET_ALL}")
            
            # Give a small delay after playback before cleanup
            await asyncio.sleep(0.5)
            
            # Now safe to cleanup
            pygame.mixer.music.unload()
            pygame.mixer.music.stop()
            
            # Clean up files after playback is complete
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"{Fore.CYAN}Cleaned up temporary file: {temp_file}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error cleaning up {temp_file}: {str(e)}{Style.RESET_ALL}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error playing audio: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.RED}Audio data type: {type(audio_data)}{Style.RESET_ALL}")
            return False

    async def play_audio(self, audio_data):
        """Play audio from bytes data asynchronously."""
        temp_files = []  # Keep track of temporary files
        try:
            # Handle base64 input
            if isinstance(audio_data, str):
                print(f"{Fore.CYAN}Decoding base64 audio data...{Style.RESET_ALL}")
                audio_data = base64.b64decode(audio_data)
            
            print(f"{Fore.CYAN}Audio data size: {len(audio_data)} bytes{Style.RESET_ALL}")
            
            # Create unique filenames using absolute paths
            temp_wav = os.path.abspath(os.path.join(os.getcwd(), f"temp_audio_{int(time.time()*1000)}.wav"))
            temp_files.append(temp_wav)
            
            # Save initial audio data
            with open(temp_wav, 'wb') as f:
                f.write(audio_data)
            print(f"{Fore.GREEN}Saved audio to {temp_wav}{Style.RESET_ALL}")
            
            # Read and convert audio if needed
            with wave.open(temp_wav, 'rb') as wav_file:
                params = wav_file.getparams()
                print(f"Input WAV parameters: {params}")
                
                # Check if conversion is needed
                needs_conversion = params.framerate != 44100 or params.nchannels != 2
                
                if needs_conversion:
                    print(f"{Fore.YELLOW}Converting audio format...{Style.RESET_ALL}")
                    
                    # Read all audio data
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    
                    # Convert sample rate if needed
                    if params.framerate != 44100:
                        print(f"Converting sample rate from {params.framerate} to 44100")
                        audio_data, _ = audioop.ratecv(audio_data, params.sampwidth, 
                                                     params.nchannels, params.framerate,
                                                     44100, None)
                    
                    # Convert to stereo if mono
                    if params.nchannels == 1:
                        print("Converting mono to stereo")
                        audio_data = audioop.tostereo(audio_data, params.sampwidth, 1, 1)
                    
                    # Create new temp file for converted audio
                    converted_wav = os.path.abspath(os.path.join(os.getcwd(), f"temp_audio_converted_{int(time.time()*1000)}.wav"))
                    temp_files.append(converted_wav)
                    
                    # Write converted audio
                    with wave.open(converted_wav, 'wb') as out_wav:
                        out_wav.setnchannels(2)
                        out_wav.setsampwidth(params.sampwidth)
                        out_wav.setframerate(44100)
                        out_wav.writeframes(audio_data)
                    
                    print(f"{Fore.GREEN}Audio conversion complete{Style.RESET_ALL}")
                    
                    # Use the converted file for playback
                    playback_file = converted_wav
                else:
                    # Use original file if no conversion needed
                    playback_file = temp_wav
            
            # Play the audio
            print(f"{Fore.CYAN}Loading audio file...{Style.RESET_ALL}")
            pygame.mixer.music.load(playback_file)
            print(f"{Fore.CYAN}Playing audio...{Style.RESET_ALL}")
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            start_time = time.time()
            duration = params.nframes / params.framerate  # Calculate audio duration
            print(f"Audio duration: {duration:.2f} seconds")
            
            while pygame.mixer.music.get_busy() or (time.time() - start_time) < duration:
                await asyncio.sleep(0.1)
            
            print(f"{Fore.GREEN}Audio playback completed{Style.RESET_ALL}")
            
            # Give a small delay after playback before cleanup
            await asyncio.sleep(0.5)
            
            # Now safe to cleanup
            pygame.mixer.music.unload()
            pygame.mixer.music.stop()
            
            # Clean up files after playback is complete
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"{Fore.CYAN}Cleaned up temporary file: {temp_file}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{Fore.RED}Error cleaning up {temp_file}: {str(e)}{Style.RESET_ALL}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Error playing audio: {str(e)}{Style.RESET_ALL}")
            print(f"{Fore.RED}Audio data type: {type(audio_data)}{Style.RESET_ALL}")
            return False

class TestClient:
    def __init__(self):
        """Initialize test client with base URL and audio recorder"""
        self.base_url = "http://localhost:8000"
        self.audio_base64 = None
        self.cleanup_temp_files()  # Clean up any leftover files from previous runs

    def cleanup_temp_files(self):
        """Clean up any temporary audio files from previous runs"""
        try:
            current_dir = os.getcwd()
            for filename in os.listdir(current_dir):
                if filename.startswith("temp_audio_") and filename.endswith(".wav"):
                    try:
                        filepath = os.path.join(current_dir, filename)
                        os.remove(filepath)
                        print(f"{Fore.CYAN}Cleaned up old temporary file: {filepath}{Style.RESET_ALL}")
                    except Exception as e:
                        print(f"{Fore.YELLOW}Could not remove old file {filepath}: {str(e)}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error during cleanup: {str(e)}{Style.RESET_ALL}")

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
                    success = asyncio.run(player.play_audio(audio_data))
                    
                    if not success:
                        print(f"{Fore.RED}Failed to play audio response{Style.RESET_ALL}")
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
        """
        Handle conversation in WebSocket connection.
        
        Processes both text and audio responses from the AI assistant.
        """
        try:
            # Send audio with return_audio flag
            message = {
                'type': 'audio',
                'audio': self.audio_base64,
                'return_audio': True  # Request audio response
            }
            
            # Send the message
            print(f"{Fore.CYAN}Sending audio data...{Style.RESET_ALL}")
            await websocket.send(json.dumps(message))
            
            # Get recognition result
            print(f"{Fore.CYAN}Waiting for recognition result...{Style.RESET_ALL}")
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("type") == "recognition":
                print(f"\n{Fore.CYAN}Recognition Result:{Style.RESET_ALL}")
                print(f"Success: {result.get('success', False)}")
                print(f"Text: {result.get('text', 'N/A')}")
                
                # Get AI response if recognition was successful
                if result.get("success"):
                    print(f"{Fore.CYAN}Waiting for AI response...{Style.RESET_ALL}")
                    response = await websocket.recv()
                    result = json.loads(response)
                    
                    if result.get("type") == "response":
                        print(f"\n{Fore.YELLOW}AI Response:{Style.RESET_ALL}")
                        print(f"Text: {result.get('text', 'N/A')}")
                        
                        # Print conversation history
                        print(f"\n{Fore.CYAN}Conversation History:{Style.RESET_ALL}")
                        for msg in result.get('history', []):
                            role = msg['role']
                            text = msg['text']
                            color = Fore.GREEN if role == 'assistant' else Fore.BLUE
                            print(f"{color}{role}: {text}{Style.RESET_ALL}")
                        
                        # Wait for audio response
                        print(f"{Fore.CYAN}Waiting for audio response...{Style.RESET_ALL}")
                        response = await websocket.recv()
                        result = json.loads(response)
                        
                        # Handle audio response
                        if result.get("type") == "audio":
                            print(f"\n{Fore.YELLOW}Audio response received{Style.RESET_ALL}")
                            audio_data = result.get("data")  # This is base64 encoded
                            if audio_data:
                                print(f"{Fore.YELLOW}Playing AI response...{Style.RESET_ALL}")
                                player = AudioPlayer()
                                success = await player.play_audio(audio_data)
                                
                                if success:
                                    print(f"{Fore.GREEN}Audio playback successful{Style.RESET_ALL}")
                                else:
                                    print(f"{Fore.RED}Failed to play audio response{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.YELLOW}No audio response received{Style.RESET_ALL}")
                            
        except websockets.exceptions.ConnectionClosed:
            print(f"{Fore.RED}WebSocket connection closed unexpectedly{Style.RESET_ALL}")
        except json.JSONDecodeError:
            print(f"{Fore.RED}Failed to parse WebSocket response{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error in WebSocket conversation: {str(e)}{Style.RESET_ALL}")

    def test_websocket_endpoint(self):
        """
        Test the /ws/voice WebSocket endpoint with audio response.
        """
        print(f"\n{Fore.CYAN}Testing WebSocket endpoint...{Style.RESET_ALL}")
        
        # Record audio first
        audio_data = self.record_audio()
        if audio_data is None:
            print(f"{Fore.RED}Failed to record audio{Style.RESET_ALL}")
            return
            
        self.audio_base64 = base64.b64encode(audio_data).decode()
        
        async def test_ws():
            uri = f"{self.base_url.replace('http', 'ws')}/ws/voice"
            try:
                async with websockets.connect(uri) as websocket:
                    await self.test_websocket_conversation(websocket)
            except websockets.exceptions.InvalidURI:
                print(f"{Fore.RED}Invalid WebSocket URI: {uri}{Style.RESET_ALL}")
            except websockets.exceptions.ConnectionRefused:
                print(f"{Fore.RED}Connection refused. Is the server running?{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}WebSocket connection error: {str(e)}{Style.RESET_ALL}")
                
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
