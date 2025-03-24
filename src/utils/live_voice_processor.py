"""
Live Voice Processor Module

This module provides integration with Google's Gemini Live API for high-quality
voice interaction capabilities.
"""

import os
import io
import base64
import logging
import asyncio
from typing import Dict, List, Any, AsyncGenerator, Optional

# Import the new Google GenAI SDK required for Gemini Live API
import google.generativeai as genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveVoiceProcessor:
    """
    Processes live voice interactions with Gemini API.
    
    This class provides methods for handling real-time voice interactions,
    including text-to-speech and speech-to-text capabilities.
    """
    
    # Available voice options
    AVAILABLE_VOICES = ["Cael", "Clover", "Ember", "Kang", "Kare", "Kori", "Kye", "Nova", "Ray", "Zuri"]
    
    def __init__(self, model_name: str = "gemini-2.0-flash-exp-image-generation", voice_name: str = "Kori"):
        """
        Initialize the Gemini voice processor.
        
        Args:
            model_name: Name of the Gemini model to use
            voice_name: Voice to use for speech responses
        """
        self.model_name = model_name
        self.voice_name = voice_name if voice_name in self.AVAILABLE_VOICES else "Kori"
        self.conversation_history = []
        
        # Initialize the client with API key
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        genai.configure(api_key=api_key)
        # Initialize Gemini model using the standard approach
        self.genai = genai
        logger.info(f"LiveVoiceProcessor initialized with model {model_name}")
        
    async def list_available_models(self) -> List[str]:
        """
        List available Gemini models.
        
        Returns:
            List of model names
        """
        try:
            # Get standard models
            standard_models = []
            for m in self.genai.list_models():
                if "gemini" in m.name.lower():
                    model_info = f"{m.name} (supported: {', '.join([g.name for g in m.supported_generation_methods])})"
                    standard_models.append(model_info)
            
            logger.info(f"Standard client models: {standard_models}")
            return standard_models
                
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return ["Error listing models"]
    
    async def connect(self) -> None:
        """
        Connect to the Gemini API and create a session.
        """
        try:
            # Create a generative model instance with conversation history support
            model = self.genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={"temperature": 0.2},
                system_instruction="You are Jarvis, an advanced AI assistant. Provide helpful, accurate, and concise responses."
            )
            
            # Create a new session
            self.session = model.start_chat(history=[])
            
            logger.info(f"Connected to Gemini API with model {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Gemini API: {str(e)}")
            raise
    
    async def process_text(self, text: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process text input and yield responses.
        
        Args:
            text: Text to process
            
        Yields:
            Dictionary containing response type and content
        """
        try:
            if not hasattr(self, 'session') or self.session is None:
                await self.connect()
            
            # Store user message in history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Send the message to the session
            response = self.session.send_message(text, stream=True)
            
            collected_text = ""
            
            # Process the response chunks
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    chunk_text = chunk.text
                    collected_text += chunk_text
                    yield {"type": "text_chunk", "content": chunk_text}
                await asyncio.sleep(0.01)  # Small delay to allow for async processing
            
            # Store the complete assistant response in history
            self.conversation_history.append({"role": "assistant", "content": collected_text})
            
            # Send complete response
            yield {"type": "text", "content": collected_text}
            yield {"type": "complete", "content": None}
            
        except Exception as e:
            logger.error(f"Error in process_text: {str(e)}")
            yield {"type": "error", "content": f"Error processing text: {str(e)}"}
            yield {"type": "complete", "content": None}

    async def process_text_with_audio(self, text: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process text input and yield responses with audio using standard API with fallback.
        
        Args:
            text: Text to process
            
        Yields:
            Dictionary containing response type and content (text, audio)
        """
        try:
            # First process the text to get the text response
            text_response = ""
            async for chunk in self.process_text(text):
                if chunk["type"] == "text":
                    text_response = chunk["content"]
                
                # Forward all chunks except complete
                if chunk["type"] != "complete":
                    yield chunk
            
            # If we have a text response, generate audio
            if text_response:
                try:
                    logger.info(f"Generating audio for text: {text_response[:50]}...")
                    
                    # Create a model for text-to-speech
                    tts_model = self.genai.GenerativeModel(
                        model_name="gemini-1.5-flash", 
                        generation_config={"temperature": 0},
                    )
                    
                    # Prepare the content with text and voice
                    tts_content = [
                        {"role": "user", "parts": [
                            {"text": f"Please convert the following text to speech using the voice '{self.voice_name}': {text_response}"}
                        ]}
                    ]
                    
                    # Generate TTS response
                    response = tts_model.generate_content(tts_content)
                    
                    # Extract audio if available
                    if hasattr(response, 'parts'):
                        for part in response.parts:
                            if hasattr(part, 'function_call'):
                                # Extract audio data if in function call response
                                if part.function_call and hasattr(part.function_call, 'args'):
                                    if hasattr(part.function_call.args, 'audio'):
                                        audio_data = part.function_call.args.audio
                                        yield {
                                            "type": "audio",
                                            "content": base64.b64encode(audio_data).decode()
                                        }
                            elif hasattr(part, 'audio'):
                                # Direct audio data
                                audio_data = part.audio
                                yield {
                                    "type": "audio",
                                    "content": base64.b64encode(audio_data).decode()
                                }
                    
                except Exception as audio_err:
                    logger.error(f"Error generating audio: {str(audio_err)}")
            
            yield {"type": "complete", "content": None}
        except Exception as e:
            logger.error(f"Error in process_text_with_audio: {str(e)}")
            yield {"type": "error", "content": f"Error processing text with audio: {str(e)}"}
            yield {"type": "complete", "content": None}
    
    async def process_audio(self, audio_data: bytes) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process audio input and yield responses with fallback method.
        
        Args:
            audio_data: Audio data to process as bytes
            
        Yields:
            Dictionary containing response type and content
        """
        try:
            # Send informational message that we're processing the audio
            yield {"type": "text", "content": "Processing your audio input..."}
            
            try:
                logger.info(f"Audio data size: {len(audio_data)} bytes")
                
                # Create a model for speech-to-text
                stt_model = self.genai.GenerativeModel("gemini-1.5-flash")
                
                # Prepare audio content
                audio_part = {"file_data": {"mime_type": "audio/webm", "file_content": audio_data}}
                audio_message = [{"role": "user", "parts": [audio_part]}]
                
                # Convert speech to text
                speech_response = stt_model.generate_content(audio_message)
                
                # Extract the transcribed text
                transcribed_text = ""
                if hasattr(speech_response, 'text'):
                    transcribed_text = speech_response.text
                elif hasattr(speech_response, 'parts'):
                    for part in speech_response.parts:
                        if hasattr(part, 'text') and part.text:
                            transcribed_text += part.text
                
                logger.info(f"Transcribed audio to: {transcribed_text}")
                
                # If transcription successful, process with text and audio
                if transcribed_text:
                    yield {"type": "text", "content": f"You said: {transcribed_text}"}
                    
                    # Process the transcribed text to get a response with audio
                    async for response in self.process_text_with_audio(transcribed_text):
                        yield response
                else:
                    yield {"type": "error", "content": "Could not transcribe audio. Please try again."}
                    yield {"type": "complete", "content": None}
                
            except Exception as process_err:
                logger.error(f"Error processing audio: {str(process_err)}")
                yield {"type": "error", "content": f"Error processing audio: {str(process_err)}"}
                yield {"type": "complete", "content": None}
            
        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            yield {"type": "error", "content": f"Error processing audio: {str(e)}"}
            yield {"type": "complete", "content": None}
    
    # Store live sessions
    live_sessions = {}
    
    async def create_live_session(self, voice_name: Optional[str] = None) -> Any:
        """
        Create a live streaming session with Gemini API.
        
        Args:
            voice_name: Optional voice name to use
        
        Returns:
            Session object
        """
        try:
            if voice_name:
                self.voice_name = voice_name if voice_name in self.AVAILABLE_VOICES else self.voice_name
            
            # Initialize the client properly
            from google import genai
            self.client = genai.Client(
                api_key=os.getenv('GOOGLE_API_KEY'),
                http_options={'api_version': 'v1alpha'}
            )
            model = self.model_name
            
            # Configure with text modality first
            # Once this works, we can add audio back
            config = {
                "response_modalities": ["TEXT"]
            }
            
            # Store the config for future use
            self.live_config = config
            self.live_model = model
            
            logger.info(f"Live session configured with model {self.model_name} and voice {self.voice_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating live session: {str(e)}", exc_info=True)
            raise
    
    async def process_live_text(self, text: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process text using live API and stream responses.
        
        Args:
            text: Text to process
            
        Yields:
            Response chunks with text and audio
        """
        try:
            logger.info(f"Starting process_live_text with input: {text[:30]}...")
            
            # Make sure we have a client configured
            if not hasattr(self, 'client') or not hasattr(self, 'live_config'):
                logger.info("Creating new live session for text processing")
                await self.create_live_session()
            
            logger.info(f"Using model: {self.live_model} with config: {self.live_config}")
            
            # Use async context manager for the live session
            logger.info("Connecting to live session...")
            async with self.client.aio.live.connect(model=self.live_model, config=self.live_config) as session:
                # Send text to the session
                logger.info(f"Sending text to session: {text[:30]}...")
                await session.send(input=text, end_of_turn=True)
                logger.info("Text sent to session, waiting for responses...")
                
                # Process responses
                response_count = 0
                async for response in session.receive():
                    response_count += 1
                    logger.info(f"Received response #{response_count} type: {type(response)}")
                    
                    if hasattr(response, 'text') and response.text is not None:
                        logger.info(f"Received text chunk: '{response.text}'")
                        yield {"type": "text_chunk", "content": response.text}
                    else:
                        logger.info(f"Received non-text response: {response}")
                    
                    # Small delay to allow for async processing
                    await asyncio.sleep(0.01)
                
                logger.info(f"Session complete, received {response_count} responses")
                # Signal completion
                yield {"type": "complete", "content": None}
            
            logger.info("Live session closed")
            
        except Exception as e:
            logger.error(f"Error in process_live_text: {str(e)}", exc_info=True)
            yield {"type": "error", "content": f"Error in live text processing: {str(e)}"}
            yield {"type": "complete", "content": None}
    
    async def process_live_audio(self, audio_data: bytes) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process audio using live API and stream responses.
        
        Args:
            audio_data: Audio data to process
            
        Yields:
            Response chunks with text
        """
        try:
            # Make sure we have a client configured
            if not hasattr(self, 'client') or not hasattr(self, 'live_config'):
                await self.create_live_session()
            
            # Use async context manager for the live session
            async with self.client.aio.live.connect(model=self.live_model, config=self.live_config) as session:
                # Send audio data to the session
                await session.send(input=audio_data, end_of_turn=True)
                
                # Process responses
                async for response in session.receive():
                    if hasattr(response, 'text') and response.text is not None:
                        logger.info(f"Received text chunk from audio: {response.text}")
                        yield {"type": "text_chunk", "content": response.text}
                    
                    # Small delay to allow for async processing
                    await asyncio.sleep(0.01)
                
                # Signal completion
                yield {"type": "complete", "content": None}
            
        except Exception as e:
            logger.error(f"Error in process_live_audio: {str(e)}")
            yield {"type": "error", "content": f"Error in live audio processing: {str(e)}"}
            yield {"type": "complete", "content": None}
            
    async def process_live_text_fallback(self, text: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Fallback implementation for processing text using standard Gemini API.
        Used when Live API is not working properly.
        
        Args:
            text: Text to process
            
        Yields:
            Response chunks with text
        """
        try:
            logger.info(f"Using fallback text processing with input: {text[:30]}...")
            
            # Configure the standard Gemini model
            if not hasattr(self, 'genai'):
                import google.generativeai as genai
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable is not set")
                genai.configure(api_key=api_key)
                self.genai = genai
            
            # Generate with the specified model and streaming
            model = self.genai.GenerativeModel(model_name=self.model_name)
            logger.info(f"Using standard Gemini API with model: {self.model_name}")
            
            # Store the prompt in history
            self.conversation_history.append({"role": "user", "content": text})
            
            # Use the non-streaming version for simplicity if streaming fails
            response = await asyncio.to_thread(model.generate_content, text)
            
            if response and hasattr(response, 'text'):
                logger.info(f"Received text response: {response.text[:50]}...")
                
                # Store in history
                self.conversation_history.append({"role": "assistant", "content": response.text})
                
                # Split into smaller chunks to simulate streaming
                words = response.text.split()
                chunk_size = 10  # Number of words per chunk
                
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i+chunk_size])
                    yield {"type": "text_chunk", "content": chunk}
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.05)
                
                # Signal completion
                yield {"type": "complete", "content": None}
            else:
                logger.error(f"Empty or invalid response from Gemini API: {response}")
                yield {"type": "error", "content": "Empty or invalid response from Gemini API"}
                yield {"type": "complete", "content": None}
                
        except Exception as e:
            logger.error(f"Error in fallback text processing: {str(e)}", exc_info=True)
            yield {"type": "error", "content": f"Error in text processing: {str(e)}"}
            yield {"type": "complete", "content": None}
    
    def get_available_voices(self) -> List[str]:
        """
        Get list of available voice options.
        
        Returns:
            List of voice names
        """
        return self.AVAILABLE_VOICES
    
    def set_voice(self, voice_name: str) -> bool:
        """
        Set the voice to use for responses.
        
        Args:
            voice_name: Name of the voice to use
            
        Returns:
            True if voice was set successfully, False otherwise
        """
        if voice_name in self.AVAILABLE_VOICES:
            self.voice_name = voice_name
            
            # Update live config if it exists
            if hasattr(self, 'live_config') and isinstance(self.live_config, dict):
                try:
                    # Update voice configuration
                    if 'speech_config' in self.live_config and 'voice_config' in self.live_config['speech_config']:
                        self.live_config['speech_config']['voice_config']['prebuilt_voice_config']['voice_name'] = voice_name
                        logger.info(f"Voice set to {voice_name} in live config")
                except Exception as e:
                    logger.error(f"Error updating voice in live config: {str(e)}")
            
            logger.info(f"Voice set to {voice_name}")
            return True
        else:
            logger.warning(f"Invalid voice name: {voice_name}")
            return False
            
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history = []
        if hasattr(self, 'session'):
            self.session = None
