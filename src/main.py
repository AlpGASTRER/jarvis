import speech_recognition as sr
import pyttsx3
import sys
import os
from utils.code_helper import CodeHelper
from utils.qa_helper import QAHelper
from dotenv import load_dotenv

class JarvisAssistant:
    def __init__(self):
        load_dotenv()
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.code_helper = CodeHelper()
        self.qa_helper = QAHelper()
        self.current_language = 'en'
        self.wit_ai_key = os.getenv('WIT_AI_KEY')
        self.setup_voice()

    def setup_voice(self):
        """Configure the text-to-speech engine"""
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        print("Voice engine initialized")

    def speak(self, text):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in speak function: {str(e)}")

    def listen(self):
        """Listen for user input through microphone"""
        try:
            with sr.Microphone() as source:
                print("\nAdjusting for ambient noise... Please wait...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print("\nListening... (Speak now)")
                
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    print("Processing speech...")
                    
                    # Try different recognition services in order
                    recognition_attempts = [
                        # First try Wit.ai if key is available
                        (lambda: self.wit_ai_key and self.recognizer.recognize_wit(audio, key=self.wit_ai_key), "Wit.ai"),
                        # Then try offline recognition
                        (lambda: self.recognizer.recognize_sphinx(audio), "Sphinx (Offline)"),
                        # Finally, try Google as last resort
                        (lambda: self.recognizer.recognize_google(audio, language=self.current_language), "Google")
                    ]
                    
                    last_error = None
                    for recognize_func, service_name in recognition_attempts:
                        try:
                            if not recognize_func:
                                continue
                                
                            text = recognize_func()
                            if text:
                                print(f"Successfully recognized with {service_name}")
                                print(f"You said: {text}")
                                return text.lower()
                        except sr.UnknownValueError:
                            last_error = "Could not understand audio"
                            continue
                        except sr.RequestError as e:
                            last_error = str(e)
                            print(f"Error with {service_name}: {e}")
                            continue
                        except Exception as e:
                            last_error = str(e)
                            print(f"Unexpected error with {service_name}: {e}")
                            continue
                    
                    # If we get here, all services failed
                    error_msg = "I'm having trouble understanding you. Please try again."
                    if "rate limit" in str(last_error).lower():
                        error_msg = "I've hit my usage limit. Please try again in a few minutes."
                    
                    print(f"All recognition services failed. Last error: {last_error}")
                    self.speak(error_msg)
                    return None
                        
                except sr.WaitTimeoutError:
                    print("No speech detected within timeout")
                    self.speak("I didn't hear anything. Please try again.")
                    return None
                    
        except Exception as e:
            print(f"Error during speech recognition: {str(e)}")
            self.speak("There was an error with the microphone. Please check your microphone settings.")
            return None

    def change_language(self, lang_code):
        """Change the assistant's language"""
        if lang_code in self.languages:
            self.current_language = lang_code
            response = f"Language changed to {self.languages[lang_code]}"
            # Always announce language change in both English and new language
            self.current_language = 'en'
            self.speak(response)
            self.current_language = lang_code
            self.speak(response)
            return True
        return False

    def is_programming_question(self, command):
        """Check if the command is related to programming"""
        programming_keywords = ["code", "programming", "function", "class", "loop", "variable", 
                              "debug", "git", "python", "javascript", "java", "error"]
        return any(keyword in command for keyword in programming_keywords)

    def process_command(self, command):
        """Process the user's command"""
        if command is None:
            return

        # Language change commands
        if "change language to" in command:
            for lang_code, lang_name in self.languages.items():
                if lang_name in command:
                    if self.change_language(lang_code):
                        return
            self.speak("Sorry, I don't support that language yet.")
            return

        # Basic commands
        if "hello" in command:
            self.speak("Hello! I'm Jarvis, your AI assistant. How can I help you today?")
        elif "goodbye" in command or "bye" in command:
            self.speak("Goodbye! Have a great day!")
            sys.exit()
        # Programming related queries
        elif self.is_programming_question(command):
            response = self.code_helper.analyze_query(command)
            self.speak(response)
        # General questions
        else:
            response = self.qa_helper.get_answer(command, self.current_language)
            self.speak(response)

    # Available languages with their codes
    languages = {
        'en': 'english',
        'ar': 'arabic',
        'fr': 'french',
        'es': 'spanish',
        'de': 'german',
        'it': 'italian',
        'ja': 'japanese',
        'ko': 'korean',
        'hi': 'hindi'
    }

def main():
    assistant = JarvisAssistant()
    assistant.speak("Hello! I'm Jarvis, your AI assistant. How can I help you?")
    
    while True:
        command = assistant.listen()
        assistant.process_command(command)

if __name__ == "__main__":
    main()
