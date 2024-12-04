import google.generativeai as genai
import os
from dotenv import load_dotenv

class CodeHelper:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key not found in environment variables")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def get_code_help(self, query, language='en'):
        """Get help for a programming-related question using Gemini"""
        try:
            # Prepare the prompt with context
            prompt = f"""You are a helpful programming assistant. Please help with this question:
            {query}"""
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1000,
                )
            )
            
            return response.text
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def generate_response(self, query):
        """Generate a response for the code-related query"""
        return self.get_code_help(query)