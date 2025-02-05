import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv
import audioop
from src.utils.enhanced_code_helper import EnhancedCodeHelper
from src.utils.audio_processor import AudioProcessor
from src.utils.voice_processor import VoiceProcessor
from src.utils.tts_processor import TTSProcessor
import google.generativeai as genai
import base64
import time
from typing import Optional

class Jarvis:
    def __init__(self):
        # Load environment variables
        self.load_env()
        
        # Initialize existing components
        self.code_helper = EnhancedCodeHelper()
        self.audio_processor = AudioProcessor()
        self.voice_processor = VoiceProcessor()
        self.tts_processor = TTSProcessor()
        
        # Initialize voice components
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Configure TTS for neutral, clear voice
        voices = self.engine.getProperty('voices')
        # Try to find a neutral voice
        neutral_voice = None
        for voice in voices:
            if 'david' in voice.name.lower() or 'mark' in voice.name.lower():
                neutral_voice = voice
                break
        if neutral_voice:
            self.engine.setProperty('voice', neutral_voice.id)
        
        # Adjust voice properties for clarity
        self.engine.setProperty('rate', 150)  # Slightly slower
        self.engine.setProperty('volume', 0.9)  # Slightly quieter
        self.engine.setProperty('pitch', 100)  # Neutral pitch
        
        # Voice recognition settings
        self.recognizer.energy_threshold = 1200
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.5
        self.recognizer.non_speaking_duration = 0.5
        
        # Initialize Gemini
        genai.configure(api_key=self.google_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.chat = self.model.start_chat(history=[])
        
        # Keep recent conversation history
        self.conversation_history = []
        self.max_history = 10
        
        # Command categories
        self.SYSTEM_COMMANDS = {
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'stop': self.cmd_exit,
            'help': self.cmd_help,
            'clear': self.cmd_clear
        }
        
        # For streaming support
        self.conversation_active = False
        self.max_retries = 3
        self.retry_delay = 0.5  # seconds
        
    def load_env(self):
        # Load environment variables
        load_dotenv()
        self.wit_ai_key = os.getenv('WIT_EN_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError("Google API key not found in environment variables")
        
    def enhance_audio(self, audio_data):
        """Process audio through the audio processor"""
        return self.audio_processor.process_audio(audio_data)

    def speak(self, text):
        """Output text via TTS and print"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def process_command(self, text):
        """Process and route commands with context"""
        text = text.lower().strip()
        
        # Add to history
        self.conversation_history.append({"role": "user", "content": text})
        self._trim_history()
        
        # Check for system commands
        first_word = text.split()[0] if text else ""
        if first_word in self.SYSTEM_COMMANDS:
            response = self.SYSTEM_COMMANDS[first_word]()
            if response != "EXIT":
                self.conversation_history.append({"role": "assistant", "content": response})
            return response
        
        # Basic greetings and responses
        if any(word in text for word in ['hi', 'hello', 'hey', 'hiya']):
            response = "Hello! I'm Jarvis, your AI assistant. How can I help you today?"
        elif 'can you hear me' in text:
            response = "Yes, I can hear you clearly! How can I help?"
        elif 'who are you' in text or "what are you" in text:
            response = "I'm Jarvis, an AI assistant powered by Google's Gemini model. I can help you with various tasks, including programming, answering questions, and general conversation."
        # Handle programming questions
        elif any(keyword in text.lower() for keyword in [
            'code', 'program', 'function', 'error', 'bug', 'python', 'javascript',
            'java', 'c++', 'debugging', 'compile', 'runtime', 'syntax'
        ]):
            response = self.handle_code_query(text)
        else:
            response = self.handle_general_query(text)
            
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def analyze_code(self, code: str, language: str = None, analysis_type: str = "full") -> dict:
        """Analyze code with specified parameters"""
        try:
            # Auto-detect language if not specified
            if not language:
                language = self.code_helper.detect_language(code)

            # Initialize response
            response = {
                "success": True,
                "language": language,
                "analysis": {},
                "suggestions": [],
                "best_practices": [],
                "security_issues": [],
                "complexity_score": 0.0
            }

            # Perform requested analysis
            if analysis_type in ["full", "syntax"]:
                syntax_result = self.code_helper.analyze_syntax(code, language)
                response["analysis"]["syntax"] = syntax_result.get("analysis", "")

            if analysis_type in ["full", "suggestions"]:
                suggestions = self.code_helper.get_suggestions(code, language)
                response["suggestions"] = suggestions

            if analysis_type in ["full", "best_practices"]:
                practices = self.code_helper.get_best_practices(code, language)
                response["best_practices"] = practices

            if analysis_type in ["full", "security"]:
                security_issues = self.code_helper.analyze_security(code, language)
                response["security_issues"] = security_issues

            # Calculate complexity
            response["complexity_score"] = self.code_helper.calculate_complexity(code, language)

            return response

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": language or "unknown"
            }

    def handle_code_query(self, text):
        """Handle programming queries with enhanced analysis"""
        # Extract code blocks if present
        code_blocks = self._extract_code_blocks(text)
        
        if code_blocks:
            # If code blocks found, analyze them
            results = []
            for code in code_blocks:
                analysis = self.analyze_code(code)
                if analysis["success"]:
                    results.append(f"Analysis for code block:\n{analysis['analysis'].get('syntax', '')}")
                    if analysis["suggestions"]:
                        results.append("\nSuggestions:\n" + "\n".join(f"- {s}" for s in analysis["suggestions"]))
            return "\n\n".join(results) if results else "I couldn't analyze the code properly."
        else:
            # If no code blocks, treat as a general programming question
            context = self._get_recent_context()
            return self.code_helper.get_code_help(text, context)

    def handle_general_query(self, text):
        """Handle general queries"""
        # Get conversation context
        context = self._get_recent_context()
        
        # Create a prompt that encourages clear, direct responses
        prompt = f"""
        Please provide a clear, direct response without markdown formatting or special characters.
        Keep the response concise and natural, as it will be spoken.
        
        User's question: {text}
        Recent context: {context if context else 'No recent context'}
        """
        
        try:
            response = self.chat.send_message(prompt).text
            # Clean up the response
            response = response.replace('*', '').replace('#', '').replace('`', '')
            response = response.replace('\n\n', ' ').replace('  ', ' ')
            return response.strip()
        except Exception as e:
            print(f"Error in general query: {str(e)}")
            return "I apologize, but I encountered an error processing your request. Could you please rephrase your question?"

    def _get_recent_context(self):
        """Get recent conversation context"""
        if len(self.conversation_history) <= 1:
            return None
            
        # Get last few messages for context
        recent = self.conversation_history[-3:-1]  # Exclude current query
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])

    def _trim_history(self):
        """Keep conversation history manageable"""
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def cmd_exit(self):
        """Handle exit command"""
        self.speak("Goodbye!")
        return "EXIT"

    def cmd_help(self):
        """Handle help command"""
        return """
        I can help you with:
        - Programming questions and code issues
        - General conversation and questions
        - System commands:
          • exit/quit/stop - End the session
          • help - Show this help message
          • clear - Clear conversation history
        
        Just speak naturally or type your question!
        """

    def cmd_clear(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.chat = self.model.start_chat(history=[])
        return "Conversation history cleared."

    def process_voice_data(self, audio_base64: str, sample_rate: int = 16000, channels: int = 1, return_audio: bool = False) -> dict:
        """Process voice data from API requests"""
        for attempt in range(self.max_retries):
            try:
                # Process voice using the voice processor
                result = self.voice_processor.process_voice(audio_base64, self.wit_ai_key, sample_rate, channels)
                
                if result["success"]:
                    # Get response for the recognized text
                    response = self.process_command(result["text"])
                    
                    response_data = {
                        "success": True,
                        "recognized_text": result["text"],
                        "response": response,
                        "audio_used": result["audio_used"]
                    }
                    
                    # Generate speech if requested
                    if return_audio:
                        tts_result = self.tts_processor.text_to_speech_base64(response)
                        if tts_result["success"]:
                            response_data["audio_response"] = tts_result
                        else:
                            response_data["audio_error"] = tts_result["error"]
                    
                    return response_data
                else:
                    if attempt == self.max_retries - 1:  # Last attempt
                        return result
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    return {
                        "success": False,
                        "error": str(e)
                    }
                time.sleep(self.retry_delay)
        
    async def process_stream(self, audio_iterator, sample_rate: int = 16000, channels: int = 1):
        """Process streaming audio data"""
        self.conversation_active = True
        buffer = b""
        chunk_size = 1024 * 16  # 16KB chunks
        
        try:
            async for audio_chunk in audio_iterator:
                if not self.conversation_active:
                    break
                    
                buffer += audio_chunk
                
                # Process when we have enough data
                if len(buffer) >= chunk_size:
                    # Convert to base64
                    audio_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Process the chunk
                    result = self.process_voice_data(audio_base64, sample_rate, channels, return_audio=True)
                    
                    if result["success"]:
                        # Stream the response audio if available
                        if "audio_response" in result and result["audio_response"]["success"]:
                            audio_data = base64.b64decode(result["audio_response"]["audio_base64"])
                            for chunk in self.tts_processor.stream_text_to_speech(result["response"]):
                                yield {
                                    "type": "audio",
                                    "data": base64.b64encode(chunk).decode('utf-8')
                                }
                        
                        # Also send the text response
                        yield {
                            "type": "text",
                            "recognized": result["recognized_text"],
                            "response": result["response"]
                        }
                    
                    buffer = b""  # Clear the buffer
                    
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e)
            }
        finally:
            self.conversation_active = False
            
    def stop_stream(self):
        """Stop the streaming conversation"""
        self.conversation_active = False

    def voice_mode(self):
        """Run in voice interaction mode"""
        self.speak("Ready for voice input. Say 'exit' to stop.")
        
        # Configure microphone
        with sr.Microphone(sample_rate=16000) as source:
            print("\nCalibrating...")
            # Longer initial calibration for better noise profile
            self.recognizer.adjust_for_ambient_noise(source, duration=5)
            print("Ready!")
            
            while True:
                try:
                    print("\nListening...")
                    # Use shorter phrase time limit to avoid processing too much audio
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                    
                    try:
                        # Try with original audio first since it seems more reliable
                        try:
                            text = self.recognizer.recognize_wit(audio, key=self.wit_ai_key)
                            print(f"✓ '{text}'")
                        except (sr.UnknownValueError, sr.RequestError) as e:
                            print(f"Original audio recognition failed, trying processed: {str(e)}")
                            # Fallback to processed audio if original fails
                            processed_audio = self.enhance_audio(audio)
                            text = self.recognizer.recognize_wit(processed_audio, key=self.wit_ai_key)
                            print(f"✓ (processed) '{text}'")
                        
                        response = self.process_command(text)
                        if response == "EXIT":
                            break
                        
                        self.speak(response)
                        
                    except sr.UnknownValueError:
                        print("× Could not understand audio")
                        continue
                    except sr.RequestError as e:
                        print(f"! Service error: {str(e)}")
                        continue
                    
                except sr.WaitTimeoutError:
                    print("No speech detected, listening again...")
                    continue
                except KeyboardInterrupt:
                    self.speak("Stopping.")
                    break
                except Exception as e:
                    print(f"Error: {str(e)}")
                    continue

    def chat_mode(self):
        """Run in chat interaction mode"""
        print("Chat mode activated. Type 'exit' to stop.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                
                response = self.process_command(user_input)
                if response == "EXIT":
                    break
                
                print(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\nStopping.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                continue

    def process_voice(self, audio_data: bytes, enhance_audio: bool = False) -> dict:
        """Process voice input and return recognized text"""
        try:
            # Create an AudioData object from the raw bytes
            audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
            
            # Enhance audio if requested
            if enhance_audio:
                audio_data = self.enhance_audio(audio_data)
                audio = sr.AudioData(audio_data, sample_rate=16000, sample_width=2)
            
            # Try Wit.ai first
            try:
                text = self.recognizer.recognize_wit(audio, key=self.wit_ai_key)
                return {
                    "success": True,
                    "text": text,
                    "audio_used": "enhanced" if enhance_audio else "original"
                }
            except sr.UnknownValueError:
                # If Wit.ai fails, try Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio)
                    return {
                        "success": True,
                        "text": text,
                        "audio_used": "enhanced" if enhance_audio else "original"
                    }
                except sr.UnknownValueError:
                    return {
                        "success": False,
                        "error": "Could not understand audio"
                    }
                except sr.RequestError as e:
                    return {
                        "success": False,
                        "error": f"Google Speech Recognition error: {str(e)}"
                    }
            except sr.RequestError as e:
                return {
                    "success": False,
                    "error": f"Wit.ai error: {str(e)}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error processing voice: {str(e)}"
            }

    def process_text(self, text: str) -> str:
        """Process text input and return AI response"""
        try:
            # Configure Gemini
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            model = genai.GenerativeModel('gemini-pro')
            
            # Get response from Gemini
            response = model.generate_content(text)
            return response.text
            
        except Exception as e:
            return f"Error processing text: {str(e)}"

    def text_to_speech(self, text: str, voice: Optional[str] = None, rate: Optional[int] = None, volume: Optional[float] = None) -> bytes:
        """Convert text to speech and return raw audio bytes"""
        try:
            # Initialize TTS processor if needed
            if not hasattr(self, 'tts_processor'):
                self.tts_processor = TTSProcessor()
            
            # Update settings if provided
            settings = {}
            if voice: settings["voice"] = voice
            if rate: settings["rate"] = rate
            if volume: settings["volume"] = volume
            
            if settings:
                self.tts_processor.update_settings(settings)
            
            # Convert text to speech
            return self.tts_processor.text_to_speech(text)
            
        except Exception as e:
            print(f"TTS error: {str(e)}")
            return None

def main():
    """Main entry point"""
    jarvis = Jarvis()
    
    print("\nSelect mode:")
    print("1. Voice")
    print("2. Chat")
    
    while True:
        try:
            choice = input("Enter choice (1/2): ").strip()
            if choice == "1":
                jarvis.voice_mode()
                break
            elif choice == "2":
                jarvis.chat_mode()
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    print("Starting Jarvis...")
    main()
