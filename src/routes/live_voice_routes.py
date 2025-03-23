"""
Live Voice Routes Module

This module implements FastAPI routes for the Gemini Live API integration in the Jarvis AI Assistant.
It provides WebSocket endpoints for real-time voice chat and configuration.

Key Features:
- WebSocket endpoint for real-time voice chat
- Text processing endpoint with Live API voice output
- Voice configuration endpoint
- Session management for Live API connections

Dependencies:
- fastapi: For API route definitions
- websockets: For WebSocket connections
- src.utils.live_voice_processor: For Live API integration
"""

import asyncio
import json
import logging
import os
import tempfile
import base64
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from src.utils.live_voice_processor import LiveVoiceProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/live", tags=["Live Voice"])

# Global voice processor instance
live_voice_processor = LiveVoiceProcessor()

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Models
class LiveTextRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    voice: Optional[str] = None

class LiveTextResponse(BaseModel):
    text: str
    audio_base64: Optional[str] = None
    session_id: str

class VoiceConfigRequest(BaseModel):
    voice: str

class VoiceConfigResponse(BaseModel):
    success: bool
    voice: str
    available_voices: List[str]

@router.post("/text", response_model=LiveTextResponse)
async def process_live_text(request: LiveTextRequest, background_tasks: BackgroundTasks):
    """
    Process text input using the Live API and return text and audio responses.
    
    Args:
        request: Text request with optional session ID and voice configuration
        
    Returns:
        Text response with audio data
    """
    # Set voice if provided
    if request.voice and request.voice != live_voice_processor.voice_name:
        live_voice_processor.set_voice(request.voice)
    
    # Process the text
    full_text = ""
    audio_chunks = []
    
    async for response in live_voice_processor.process_text(request.text):
        if response["type"] == "text":
            full_text += response["content"]
        elif response["type"] == "audio":
            audio_chunks.append(response["content"])
    
    # Create a temporary file for the audio
    audio_base64 = None
    if audio_chunks:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Save audio to file
        await live_voice_processor.save_audio_to_file(audio_chunks, temp_filename)
        
        # Read the file and encode as base64
        with open(temp_filename, "rb") as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        
        # Schedule file deletion
        background_tasks.add_task(os.unlink, temp_filename)
    
    # Generate a session ID if not provided
    session_id = request.session_id or "session_" + str(id(live_voice_processor))
    
    return LiveTextResponse(
        text=full_text,
        audio_base64=audio_base64,
        session_id=session_id
    )

@router.post("/voice-config", response_model=VoiceConfigResponse)
async def configure_voice(request: VoiceConfigRequest):
    """
    Configure the voice for the Live API.
    
    Args:
        request: Voice configuration request
        
    Returns:
        Voice configuration response
    """
    success = live_voice_processor.set_voice(request.voice)
    
    return VoiceConfigResponse(
        success=success,
        voice=live_voice_processor.voice_name,
        available_voices=live_voice_processor.get_available_voices()
    )

@router.get("/voices", response_model=List[str])
async def get_available_voices():
    """
    Get the list of available voices.
    
    Returns:
        List of available voice names
    """
    return live_voice_processor.get_available_voices()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time voice chat.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        # Create a new session
        await live_voice_processor.create_session()
        
        # Send initial connection message
        await websocket.send_json({
            "type": "connection_established",
            "client_id": client_id,
            "voice": live_voice_processor.voice_name
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive()
            
            if "text" in data:
                # Process text input
                message = data["text"]
                
                async for response in live_voice_processor.process_text(message):
                    if response["type"] == "text":
                        await websocket.send_json({
                            "type": "text",
                            "content": response["content"]
                        })
                    elif response["type"] == "audio":
                        # Convert binary audio data to base64 for WebSocket transmission
                        audio_base64 = base64.b64encode(response["content"]).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio",
                            "content": audio_base64
                        })
                    elif response["type"] == "complete":
                        await websocket.send_json({
                            "type": "complete"
                        })
            
            elif "audio" in data:
                # Process audio input
                audio_data = base64.b64decode(data["audio"])
                
                async for response in live_voice_processor.process_audio(audio_data):
                    if response["type"] == "text":
                        await websocket.send_json({
                            "type": "text",
                            "content": response["content"]
                        })
                    elif response["type"] == "audio":
                        # Convert binary audio data to base64 for WebSocket transmission
                        audio_base64 = base64.b64encode(response["content"]).decode("utf-8")
                        await websocket.send_json({
                            "type": "audio",
                            "content": audio_base64
                        })
                    elif response["type"] == "complete":
                        await websocket.send_json({
                            "type": "complete"
                        })
            
            elif "config" in data:
                # Handle configuration updates
                config = data["config"]
                
                if "voice" in config:
                    success = live_voice_processor.set_voice(config["voice"])
                    await websocket.send_json({
                        "type": "config_update",
                        "voice": live_voice_processor.voice_name,
                        "success": success
                    })
                
                if "clear_history" in config and config["clear_history"]:
                    live_voice_processor.clear_conversation_history()
                    await websocket.send_json({
                        "type": "history_cleared"
                    })
    
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up
        if client_id in active_connections:
            del active_connections[client_id]
        await live_voice_processor.close_session()

@router.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources when the application shuts down.
    """
    await live_voice_processor.close_session()
