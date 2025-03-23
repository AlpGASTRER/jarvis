"""
Live Voice Processor Module

This module implements voice interaction using the Gemini Live API for the Jarvis AI Assistant.
It provides real-time voice chat capabilities with high-quality text-to-speech output.

Key Features:
- Real-time audio streaming with Gemini Live API
- High-quality voice output with configurable voices
- Seamless conversation handling with context management
- Support for interruptions and natural conversation flow

Dependencies:
- google.generativeai: For Gemini Live API integration
- asyncio: For asynchronous processing
- numpy: For audio data manipulation
- wave: For WAV file handling
"""

import os
import asyncio
import numpy as np
import wave
import base64
import logging
import tempfile
from typing import Union, Tuple, List, Dict, Any, Optional, AsyncGenerator
import google.generativeai as genai
from google.generativeai.types import (
    LiveConnectConfig, 
    SpeechConfig, 
    VoiceConfig, 
    PrebuiltVoiceConfig,
    Content,
    Part
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveVoiceProcessor:
    """
    A class for processing voice input and output using the Gemini Live API.
    
    This class handles real-time voice interactions with the Gemini model,
    providing high-quality voice output and natural conversation capabilities.
    
    Attributes:
        model_name: Name of the Gemini model to use
        voice_name: Name of the voice to use for TTS
        client: Gemini API client
        active_session: Current active Live API session
        conversation_history: List of conversation turns
    """
    
    AVAILABLE_VOICES = ["Aoede", "Charon", "Fenrir", "Kore", "Puck"]
    
    def __init__(self, model_name: str = "gemini-2.0-pro-exp-02-05", voice_name: str = "Kore"):
        """
        Initialize the Live Voice Processor.
        
        Args:
            model_name: Name of the Gemini model to use
            voice_name: Name of the voice to use for TTS (one of Aoede, Charon, Fenrir, Kore, Puck)
        """
        self.model_name = model_name
        self.voice_name = voice_name if voice_name in self.AVAILABLE_VOICES else "Kore"
        
        # Initialize Gemini API client
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.client = genai.Client(http_options={'api_version': 'v1alpha'})
        
        # Initialize session and conversation history
        self.active_session = None
        self.conversation_history = []
        
        # Audio configuration
        self.input_sample_rate = 16000  # 16kHz for input
        self.output_sample_rate = 24000  # 24kHz for output
        
        logger.info(f"LiveVoiceProcessor initialized with model {model_name} and voice {voice_name}")
    
    async def create_session(self) -> None:
        """
        Create a new Live API session with the configured model and voice.
        """
        if self.active_session:
            await self.close_session()
        
        # Configure the session for both text and audio responses
        config = LiveConnectConfig(
            response_modalities=["TEXT", "AUDIO"],
            speech_config=SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name=self.voice_name)
                )
            )
        )
        
        # Create the session
        self.active_session = await self.client.aio.live.connect(
            model=self.model_name, 
            config=config
        )
        
        logger.info(f"Created new Live API session with model {self.model_name}")
        
        # If we have conversation history, send it to the model
        if self.conversation_history:
            await self._send_conversation_history()
    
    async def close_session(self) -> None:
        """
        Close the active Live API session.
        """
        if self.active_session:
            await self.active_session.close()
            self.active_session = None
            logger.info("Closed Live API session")
    
    async def _send_conversation_history(self) -> None:
        """
        Send the conversation history to the model to establish context.
        """
        if not self.active_session or not self.conversation_history:
            return
        
        # Create a list of turns from the conversation history
        turns = []
        for turn in self.conversation_history:
            turns.append(Content(
                parts=[Part(text=turn["text"])],
                role=turn["role"]
            ))
        
        # Send the turns to the model
        await self.active_session.send(
            input=genai.types.LiveClientContent(turns=turns)
        )
        
        logger.info(f"Sent conversation history with {len(turns)} turns to the model")
    
    async def process_text(self, text: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process text input and yield text and audio responses.
        
        Args:
            text: Text input to process
            
        Yields:
            Dictionary containing response type and content
        """
        if not self.active_session:
            await self.create_session()
        
        # Add the user's message to the conversation history
        self.conversation_history.append({
            "text": text,
            "role": "user"
        })
        
        # Send the message to the model
        await self.active_session.send(
            input=text,
            end_of_turn=True
        )
        
        # Process the response
        model_response_text = ""
        audio_chunks = []
        
        async for response in self.active_session.receive():
            # Handle text response
            if response.text is not None:
                model_response_text += response.text
                yield {
                    "type": "text",
                    "content": response.text
                }
            
            # Handle audio response
            if response.data is not None:
                audio_chunks.append(response.data)
                yield {
                    "type": "audio",
                    "content": response.data
                }
        
        # Add the model's response to the conversation history
        if model_response_text:
            self.conversation_history.append({
                "text": model_response_text,
                "role": "model"
            })
            
        # Return a completion event
        yield {
            "type": "complete",
            "content": None
        }
    
    async def process_audio(self, audio_data: bytes) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process audio input and yield text and audio responses.
        
        Args:
            audio_data: Raw audio data (16-bit PCM, 16kHz, mono)
            
        Yields:
            Dictionary containing response type and content
        """
        if not self.active_session:
            await self.create_session()
        
        # Send the audio data to the model
        await self.active_session.send(
            input=audio_data,
            end_of_turn=True
        )
        
        # Process the response
        model_response_text = ""
        audio_chunks = []
        
        async for response in self.active_session.receive():
            # Handle text response
            if response.text is not None:
                model_response_text += response.text
                yield {
                    "type": "text",
                    "content": response.text
                }
            
            # Handle audio response
            if response.data is not None:
                audio_chunks.append(response.data)
                yield {
                    "type": "audio",
                    "content": response.data
                }
        
        # Add the conversation to history if we got a text response
        if model_response_text:
            # We don't have the original text from audio, so we'll just note it was audio
            self.conversation_history.append({
                "text": "[Audio Input]",
                "role": "user"
            })
            self.conversation_history.append({
                "text": model_response_text,
                "role": "model"
            })
            
        # Return a completion event
        yield {
            "type": "complete",
            "content": None
        }
    
    async def save_audio_to_file(self, audio_chunks: List[bytes], filename: str) -> str:
        """
        Save audio chunks to a WAV file.
        
        Args:
            audio_chunks: List of audio data chunks
            filename: Name of the file to save
            
        Returns:
            Path to the saved file
        """
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.output_sample_rate)  # 24kHz
            
            for chunk in audio_chunks:
                wf.writeframes(chunk)
        
        return filename
    
    def get_available_voices(self) -> List[str]:
        """
        Get the list of available voices.
        
        Returns:
            List of available voice names
        """
        return self.AVAILABLE_VOICES
    
    def set_voice(self, voice_name: str) -> bool:
        """
        Set the voice to use for TTS.
        
        Args:
            voice_name: Name of the voice to use
            
        Returns:
            True if the voice was set successfully, False otherwise
        """
        if voice_name in self.AVAILABLE_VOICES:
            self.voice_name = voice_name
            return True
        return False
    
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history = []
        logger.info("Cleared conversation history")
