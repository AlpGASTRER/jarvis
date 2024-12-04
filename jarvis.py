import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv
import audioop
from src.utils.code_helper import CodeHelper
import google.generativeai as genai

class Jarvis:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.wit_ai_key = os.getenv('WIT_AI_KEY')
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize voice components
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Voice recognition settings
        self.recognizer.energy_threshold = 1200
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.5
        self.recognizer.non_speaking_duration = 0.5
        
        # Configure TTS
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 1.0)
        
        # Initialize code helper
        self.code_helper = CodeHelper()
        
        # Initialize conversation history
        self.conversation_history = [
            {"role": "system", "content": "You are Jarvis, a helpful AI assistant. You are knowledgeable, friendly, and concise in your responses."}
        ]
        
        # Command categories
        self.SYSTEM_COMMANDS = {
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'stop': self.cmd_exit,
            'help': self.cmd_help,
            'clear': self.cmd_clear
        }

    def enhance_audio(self, audio_data):
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

    def speak(self, text):
        """Output text via TTS and print"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def process_command(self, text):
        """Process and route commands to appropriate handlers"""
        text = text.lower().strip()
        
        # Check for system commands first
        first_word = text.split()[0] if text else ""
        if first_word in self.SYSTEM_COMMANDS:
            return self.SYSTEM_COMMANDS[first_word]()
            
        # Check if it's a programming question
        if any(keyword in text.lower() for keyword in [
            'code', 'program', 'function', 'error', 'bug', 'python', 'javascript',
            'java', 'c++', 'debugging', 'compile', 'runtime', 'syntax'
        ]):
            return self.handle_code_query(text)
            
        # General conversation
        return self.handle_general_query(text)

    def handle_code_query(self, text):
        """Handle programming-related queries"""
        try:
            return self.code_helper.get_code_help(text)
        except Exception as e:
            return f"Sorry, I encountered an error processing your code query: {str(e)}"

    def handle_general_query(self, text):
        """Handle general conversation using Gemini"""
        try:
            # Add user message to history
            self.conversation_history.append(f"Human: {text}")
            
            # Build conversation context
            context = "\n".join(self.conversation_history[-5:])  # Last 5 messages
            prompt = f"""Previous conversation:
            {context}
            
            You are Jarvis, a helpful AI assistant. Be knowledgeable, friendly, and concise.
            Please respond to the last message."""
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=150,
                )
            )
            
            # Extract and store response
            ai_response = response.text
            self.conversation_history.append(f"Assistant: {ai_response}")
            
            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return ai_response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

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
        self.conversation_history = [self.conversation_history[0]]  # Keep only system message
        return "Conversation history cleared."

    def voice_mode(self):
        """Run in voice interaction mode"""
        self.speak("Ready for voice input. Say 'exit' to stop.")
        
        with sr.Microphone() as source:
            print("\nCalibrating...")
            self.recognizer.adjust_for_ambient_noise(source, duration=3)
            print("Ready!")
            
            while True:
                try:
                    print("\nListening...")
                    audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=None)
                    
                    try:
                        text = self.recognizer.recognize_wit(self.enhance_audio(audio), key=self.wit_ai_key)
                        print(f"✓ '{text}'")
                        
                        response = self.process_command(text)
                        if response == "EXIT":
                            break
                        
                        self.speak(response)
                        
                    except sr.UnknownValueError:
                        print("×")
                    except sr.RequestError:
                        print("!")
                        
                except sr.WaitTimeoutError:
                    continue
                except KeyboardInterrupt:
                    self.speak("Stopping.")
                    break
                except:
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
                print("\nStopping chat mode.")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    jarvis = Jarvis()
    
    # Ask for mode preference
    while True:
        mode = input("Choose mode (voice/chat): ").lower().strip()
        if mode == 'voice':
            jarvis.voice_mode()
            break
        elif mode == 'chat':
            jarvis.chat_mode()
            break
        else:
            print("Please choose 'voice' or 'chat'")

if __name__ == "__main__":
    print("Starting Jarvis...")
    main()
