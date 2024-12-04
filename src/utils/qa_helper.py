import os
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

class QAHelper:
    def __init__(self):
        # Initialize OpenAI client with API key from environment variable
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.openai.com/v1"  # Explicitly set the base URL
        )
        self.max_retries = 3
        self.retry_delay = 60  # seconds

    def get_answer(self, question, language='en'):
        """Get answer for a general question using GPT"""
        prompt = f"Answer the following question in {language}: {question}"
        response = self.get_response(prompt)
        return response

    def get_response(self, question):
        """Get response from OpenAI with rate limit handling"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides clear and concise answers."},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()
            
            except Exception as e:
                error_str = str(e).lower()
                if "rate limit" in error_str:
                    if attempt < self.max_retries - 1:
                        print(f"Rate limit hit. Waiting {self.retry_delay} seconds before retry...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return "I apologize, but I've hit my usage limit. Please try again in a few minutes."
                elif "quota" in error_str:
                    return "I apologize, but I've reached my quota limit. Please try again later or contact support."
                else:
                    return f"An error occurred: {str(e)}"
        
        return "I'm having trouble accessing the OpenAI service right now. Please try again later."
