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
        
        # Initialize model with focused settings for code
        generation_config = genai.types.GenerationConfig(
            temperature=0.3,  # Lower temperature for more precise code responses
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,  # Increased for detailed code explanations
        )
        
        self.model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)
        self.chat = self.model.start_chat(history=[])
        
        # Initialize with coding context
        self.chat.send_message("""You are an expert programming assistant. 
        Provide clear, concise code explanations and examples. 
        Focus on best practices and efficient solutions.""")

    def get_code_help(self, query, context=None):
        """Get help for programming questions with optional context"""
        try:
            # Prepare prompt with context
            prompt = self._prepare_prompt(query, context)
            
            # Generate response
            response = self.chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
            
    def _prepare_prompt(self, query, context=None):
        """Prepare a focused prompt for code-related queries"""
        if context:
            return f"""Context: {context}
            
            Question: {query}
            
            Please provide a clear explanation with code examples if relevant."""
        
        return f"""Question: {query}
        
        Please provide a clear explanation with code examples if relevant."""