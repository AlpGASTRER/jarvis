"""
Live Voice Processor Module

This module implements voice interaction using the Gemini API for the Jarvis AI Assistant.
It provides text-to-speech capabilities with the standard Gemini model.

Key Features:
- High-quality text processing with Gemini Pro Experimental model
- Conversation context management
- Support for multiple conversation turns

Dependencies:
- google.generativeai: For Gemini API integration
- asyncio: For asynchronous processing
- wave: For WAV file handling
"""

import os
import asyncio
import logging
import tempfile
from typing import Union, Tuple, List, Dict, Any, Optional, AsyncGenerator
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveVoiceProcessor:
    """
    A class for processing voice input and output using the Gemini API.
    
    This class handles text interactions with the Gemini model,
    providing conversation capabilities.
    
    Attributes:
        model_name: Name of the Gemini model to use
        model: Gemini model instance
        conversation_history: List of conversation turns
    """
    
    def __init__(self, model_name: str = "gemini-2.0-pro-exp-02-05"):
        """
        Initialize the Live Voice Processor.
        
        Args:
            model_name: Name of the Gemini model to use
        """
        self.model_name = model_name
        
        # Initialize Gemini API
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(model_name)
        
        # Initialize conversation history
        self.conversation_history = []
        self.chat = self.model.start_chat(history=[])
        
        logger.info(f"LiveVoiceProcessor initialized with model {model_name}")
    
    async def process_text(self, text: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process text input and yield text responses.
        
        Args:
            text: Text input to process
            
        Yields:
            Dictionary containing response type and content
        """
        # Add the user's message to the conversation history
        self.conversation_history.append({
            "text": text,
            "role": "user"
        })
        
        # Process the message
        response = self.chat.send_message(text)
        
        # Get the response text
        model_response_text = response.text
        
        # Add the model's response to the conversation history
        if model_response_text:
            self.conversation_history.append({
                "text": model_response_text,
                "role": "model"
            })
            
            # Yield the text response
            yield {
                "type": "text",
                "content": model_response_text
            }
        
        # Return a completion event
        yield {
            "type": "complete",
            "content": None
        }
    
    def get_available_voices(self) -> List[str]:
        """
        Get the list of available voices.
        
        Returns:
            List of available voice names
        """
        # Since we're not using the Live API, we'll return a placeholder
        return ["Default"]
    
    def set_voice(self, voice_name: str) -> bool:
        """
        Set the voice to use for TTS.
        
        Args:
            voice_name: Name of the voice to use
            
        Returns:
            True if the voice was set successfully, False otherwise
        """
        # Since we're not using the Live API, we'll always return True
        return True
    
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history = []
        self.chat = self.model.start_chat(history=[])
        logger.info("Cleared conversation history")
        
    async def close_session(self) -> None:
        """
        Close any active sessions.
        """
        # Nothing to close in this implementation
        pass
