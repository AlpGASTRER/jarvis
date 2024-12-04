import google.generativeai as genai
import os
from dotenv import load_dotenv

class CodeHelper:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Use the most capable model
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Set up the chat
        self.chat = self.model.start_chat(history=[])

    def get_code_help(self, query, language='en'):
        """Get help for a programming-related question using Gemini"""
        try:
            # Add programming context to the query
            prompt = f"""As a programming assistant, help with this question. 
            Include code examples if relevant.
            
            Question: {query}"""
            
            # Generate response
            response = self.chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def generate_response(self, query):
        """Generate a response for the code-related query"""
        return self.get_code_help(query)