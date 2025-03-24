"""
Live Voice Routes Module

This module implements FastAPI routes for voice interaction with the Jarvis AI Assistant.
It provides endpoints for processing text and audio input with Gemini responses.

Key Features:
- Text and audio processing with Gemini API
- WebSocket support for real-time interaction
- Conversation history management
- Voice configuration options
"""

import os
import asyncio
import json
import logging
import base64
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.utils.live_voice_processor import LiveVoiceProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/live", tags=["Live Voice"])

# Create voice processor
voice_processor = LiveVoiceProcessor()

# Store active connections
active_connections: Dict[str, WebSocket] = {}

class TextRequest(BaseModel):
    """Request model for text processing."""
    text: str
    voice: Optional[str] = "Kore"

class TextResponse(BaseModel):
    """Response model for text processing."""
    text: str
    audio_base64: Optional[str] = None

class AudioRequest(BaseModel):
    """Request model for audio processing."""
    audio_base64: str
    voice: Optional[str] = "Kore"

@router.post("/text", response_model=TextResponse)
async def process_text(request: TextRequest, background_tasks: BackgroundTasks):
    """
    Process text input and return a response with audio.
    
    Args:
        request: TextRequest object containing text to process
        
    Returns:
        TextResponse object containing the response text and audio
    """
    try:
        # Set voice if specified
        if request.voice:
            voice_processor.set_voice(request.voice)
        
        # Process text with audio using the Live API
        response_text = ""
        audio_data = None
        
        async for response in voice_processor.process_text_with_audio(request.text):
            if response["type"] == "text":
                response_text = response["content"]
            elif response["type"] == "text_chunk":
                # For streaming, we'll just collect the final text at the end
                continue
            elif response["type"] == "audio":
                audio_data = response["content"]  # Base64 encoded audio
            elif response["type"] == "error":
                raise HTTPException(status_code=500, detail=response["content"])
        
        return TextResponse(text=response_text, audio_base64=audio_data)
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@router.post("/audio", response_model=TextResponse)
async def process_audio(request: AudioRequest, background_tasks: BackgroundTasks):
    """
    Process audio input and return a response with audio.
    
    Args:
        request: AudioRequest object containing audio to process
        
    Returns:
        TextResponse object containing the response text and audio
    """
    try:
        # Set voice if specified
        if request.voice:
            voice_processor.set_voice(request.voice)
        
        # Decode the audio
        audio_data = base64.b64decode(request.audio_base64)
        
        # Process audio with Live API
        response_text = ""
        response_audio = None
        
        async for response in voice_processor.process_audio(audio_data):
            if response["type"] == "text":
                response_text = response["content"]
            elif response["type"] == "text_chunk":
                # For streaming, we'll just collect the final text
                continue
            elif response["type"] == "audio":
                response_audio = response["content"]  # Base64 encoded audio
            elif response["type"] == "error":
                raise HTTPException(status_code=500, detail=response["content"])
        
        return TextResponse(text=response_text, audio_base64=response_audio)
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time voice interaction.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    logger.info(f"New WebSocket connection from client: {client_id}")
    await websocket.accept()
    active_connections[client_id] = websocket
    logger.info(f"Client {client_id} connected successfully")
    
    try:
        # Send initial connection message
        connection_msg = {
            "type": "connection_established",
            "voice": voice_processor.voice_name,
            "client_id": client_id,
            "available_voices": voice_processor.get_available_voices()
        }
        logger.info(f"Sending connection info to client {client_id}: {connection_msg}")
        await websocket.send_json(connection_msg)
        
        while True:
            # Receive message from client
            logger.info(f"Waiting for message from client {client_id}")
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"Received message from client {client_id}: {message}")
            
            # Handle text input
            if "text" in message:
                text = message["text"]
                logger.info(f"Processing text from client {client_id}: {text}")
                
                try:
                    # First try with Live API
                    logger.info(f"Attempting to process text with Live API: {text}")
                    try_fallback = False
                    response_count = 0
                    try:
                        async for response in voice_processor.process_live_text(text):
                            response_count += 1
                            if response["type"] == "error":
                                logger.warning(f"Error in Live API: {response['content']}")
                                try_fallback = True
                                break
                            logger.info(f"Sending response #{response_count} to client {client_id}: {response}")
                            await websocket.send_json(response)
                    except Exception as e:
                        logger.error(f"Exception in Live API: {str(e)}", exc_info=True)
                        try_fallback = True
                    
                    # If Live API failed, try fallback
                    if try_fallback or response_count == 0:
                        logger.info(f"Falling back to standard API for client {client_id}")
                        response_count = 0
                        async for response in voice_processor.process_live_text_fallback(text):
                            response_count += 1
                            logger.info(f"Sending fallback response #{response_count} to client {client_id}: {response}")
                            await websocket.send_json(response)
                    
                    logger.info(f"Completed processing for client {client_id}, sent {response_count} responses")
                except Exception as e:
                    logger.error(f"Error processing text for client {client_id}: {str(e)}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error processing text: {str(e)}"
                    })
                    await websocket.send_json({
                        "type": "complete",
                        "content": None
                    })
            
            # Handle voice configuration
            elif "set_voice" in message:
                voice_name = message["set_voice"]
                logger.info(f"Setting voice for client {client_id} to: {voice_name}")
                success = voice_processor.set_voice(voice_name)
                result_msg = {
                    "type": "voice_changed" if success else "error",
                    "content": f"Voice set to {voice_name}" if success else f"Invalid voice: {voice_name}"
                }
                logger.info(f"Voice change result for client {client_id}: {result_msg}")
                await websocket.send_json(result_msg)
            
            # Handle audio input
            elif "audio" in message:
                try:
                    # Decode base64 audio data
                    audio_data = base64.b64decode(message["audio"])
                    logger.info(f"Received audio from client {client_id}, {len(audio_data)} bytes")
                    
                    # Process audio with Live API
                    async for response in voice_processor.process_live_audio(audio_data):
                        await websocket.send_json(response)
                except Exception as e:
                    logger.error(f"Error processing audio data: {str(e)}")
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error processing audio: {str(e)}"
                    })
            
            # Handle history clear
            elif "clear_history" in message and message["clear_history"]:
                voice_processor.clear_conversation_history()
                await websocket.send_json({
                    "type": "history_cleared",
                    "content": "Conversation history has been cleared"
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "content": "Unknown message type"
                })
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        if client_id in active_connections:
            del active_connections[client_id]
    except Exception as e:
        logger.error(f"Error in websocket connection: {str(e)}")
        if client_id in active_connections:
            del active_connections[client_id]

@router.get("/voices")
async def get_available_voices():
    """
    Get the list of available voices.
    
    Returns:
        List of available voice names
    """
    try:
        voices = voice_processor.get_available_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error getting available voices: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting available voices: {str(e)}")
