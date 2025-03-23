"""
Live Voice Routes Module

This module implements FastAPI routes for voice interaction using the Gemini API.
It provides endpoints for processing text input with Gemini responses.

Key Features:
- Text processing with Gemini API
- WebSocket support for real-time interaction
- Conversation history management
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
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
    voice: Optional[str] = "Default"

class TextResponse(BaseModel):
    """Response model for text processing."""
    text: str

@router.post("/text", response_model=TextResponse)
async def process_text(request: TextRequest):
    """
    Process text input and return a response.
    
    Args:
        request: TextRequest object containing text to process
        
    Returns:
        TextResponse object containing the response text
    """
    try:
        # Set voice if specified
        if request.voice:
            voice_processor.set_voice(request.voice)
        
        # Process text
        response_text = ""
        async for response in voice_processor.process_text(request.text):
            if response["type"] == "text":
                response_text += response["content"]
        
        return TextResponse(text=response_text)
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time voice interaction.
    
    Args:
        websocket: WebSocket connection
        client_id: Client identifier
    """
    await websocket.accept()
    active_connections[client_id] = websocket
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection_established",
            "voice": "Default",
            "client_id": client_id
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle text input
            if "text" in message:
                text = message["text"]
                logger.info(f"Received text from client {client_id}: {text}")
                
                # Process text
                async for response in voice_processor.process_text(text):
                    if response["type"] == "text":
                        await websocket.send_json({
                            "type": "text",
                            "content": response["content"]
                        })
                    elif response["type"] == "complete":
                        await websocket.send_json({
                            "type": "complete",
                            "content": None
                        })
            
            # Handle configuration updates
            elif "config" in message:
                config = message["config"]
                
                # Handle voice change
                if "voice" in config:
                    voice = config["voice"]
                    success = voice_processor.set_voice(voice)
                    await websocket.send_json({
                        "type": "config_update",
                        "voice": voice,
                        "success": success
                    })
                
                # Handle history clearing
                if "clear_history" in config and config["clear_history"]:
                    voice_processor.clear_conversation_history()
                    await websocket.send_json({
                        "type": "config_update",
                        "clear_history": True,
                        "success": True
                    })
    
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
        if client_id in active_connections:
            del active_connections[client_id]
    
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {str(e)}")
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
