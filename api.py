"""
Jarvis AI Assistant API

This module implements the FastAPI-based REST API for the Jarvis AI Assistant.
It provides endpoints for text processing, voice recognition, code analysis,
and real-time voice interaction through WebSocket.

Key Features:
- Text processing with optional TTS
- Voice recognition with noise reduction
- Multi-language code analysis
- Real-time voice interaction
- Conversation management
- Customizable TTS voices
- Gemini Live API integration for high-quality voice chat

Dependencies:
- FastAPI: Web framework
- Pydantic: Data validation
- uvicorn: ASGI server
- Jarvis: Core AI assistant functionality
"""

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, Query, Depends, Header, WebSocketDisconnect, File, Form, UploadFile, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import json
import asyncio
import time
import io
import wave
import base64
from src.utils.enhanced_code_helper import EnhancedCodeHelper
from src.utils.voice_processor import VoiceProcessor
from src.routes.live_voice_routes import router as live_voice_router
import os
import logging
import platform

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize core components
code_helper = EnhancedCodeHelper()

# Initialize FastAPI with metadata
app = FastAPI(
    title="Jarvis AI Assistant API",
    description="API for voice interaction with AI assistant",
    version="1.0.0"
)

# Configure CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Update this in production with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the Live Voice API routes
app.include_router(live_voice_router)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize processors
voice_processor = VoiceProcessor()
code_helper = EnhancedCodeHelper()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception:
                    pass

manager = ConnectionManager()

# Request/Response Models
class TextRequest(BaseModel):
    """
    Model for text processing requests.
    
    Attributes:
        text: Input text to process
        mode: Processing mode (general/code/voice)
        return_audio: Whether to return TTS audio
        voice_settings: Custom TTS voice configuration
    """
    text: str = Field(..., description="The text input to process")
    mode: Optional[str] = Field("general", description="Processing mode: 'general', 'code', or 'voice'")
    return_audio: Optional[bool] = Field(False, description="Whether to return audio response")
    voice_settings: Optional[Dict[str, Any]] = Field(None, description="Custom voice settings (rate, volume, voice)")

class AudioRequest(BaseModel):
    """
    Model for audio processing requests.
    
    Attributes:
        audio_base64: Base64 encoded audio data
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels
        return_audio: Whether to return TTS audio
        enhance_audio: Whether to apply noise reduction
    """
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    channels: int = Field(1, description="Number of audio channels")
    return_audio: Optional[bool] = Field(False, description="Whether to return audio response")
    enhance_audio: Optional[bool] = Field(True, description="Whether to apply noise reduction")

class CodeMetrics(BaseModel):
    """
    Model for code metrics data.
    
    Attributes:
        total_lines: Total number of code lines
        non_empty_lines: Number of non-empty lines
        average_line_length: Average length of code lines
        num_functions: Number of functions (if available)
        num_classes: Number of classes (if available)
        num_imports: Number of imports (if available)
        cyclomatic_complexity: Cyclomatic complexity score
    """
    total_lines: int
    non_empty_lines: int
    average_line_length: float
    num_functions: Optional[int] = None
    num_classes: Optional[int] = None
    num_imports: Optional[int] = None
    cyclomatic_complexity: Optional[float] = None

class CodeRequest(BaseModel):
    """Code analysis request model"""
    code: str
    language: str
    analysis_type: str = "full"  # Default to full analysis

class CodeResponse(BaseModel):
    """
    Model for code analysis responses.
    
    Attributes:
        success: Whether analysis was successful
        language: Detected/specified language
        analysis: Detailed analysis results
        suggestions: Code improvement suggestions
        best_practices: Best practices recommendations
        security_issues: Security vulnerabilities found
        complexity_score: Code complexity metric
        metrics: Detailed code metrics
        error: Error message if analysis failed
    """
    success: bool
    language: str
    analysis: Dict[str, Any]
    suggestions: List[str]
    best_practices: Optional[List[str]]
    security_issues: Optional[List[str]]
    complexity_score: Optional[float]
    metrics: Optional[Dict[str, Any]]
    error: Optional[str] = None

class TextResponse(BaseModel):
    """
    Model for text processing responses.
    
    Attributes:
        success: Whether processing was successful
        response: Text response
        audio_response: Optional audio response
    """
    success: bool
    response: str
    audio_response: Optional[Dict[str, Any]] = None

class ConversationRequest(BaseModel):
    """Request model for conversation management."""
    action: str = Field(..., description="Action to perform: start, continue, clear, list, switch")
    text: Optional[str] = Field(None, description="Text to process")
    chat_id: Optional[str] = Field(None, description="Chat ID to use")

# API Routes
@app.post("/text", response_model=TextResponse, tags=["Text"])
async def process_text(request: TextRequest):
    """
    Process text input and get AI response.
    Supports general queries, code-related questions, and voice commands.
    Can return both text and audio responses.
    """
    try:
        # Process based on mode
        if request.mode == "code":
            # Use code helper for code-related queries
            response = code_helper.get_code_help(request.text)
        else:
            # Get general AI response with history
            response = voice_processor.get_ai_response(request.text)
            
        result = {
            "success": True,
            "response": response,
            "audio_response": None,
            "history": voice_processor.get_history()
        }
        
        # Generate audio response if requested
        if request.return_audio:
            audio_data = voice_processor.text_to_speech(
                response, 
                voice_settings=request.voice_settings
            )
            if audio_data:
                result["audio_response"] = {
                    "audio_base64": base64.b64encode(audio_data).decode(),
                    "sample_rate": 22050,
                    "channels": 1
                }
                
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice", tags=["Voice"])
async def process_voice(
    audio_file: UploadFile = File(..., description="Audio file to process"),
    enhance_audio: bool = Form(False, description="Whether to apply audio enhancement")
):
    """Process voice input and return AI response"""
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Initialize voice processor
        from src.utils.voice_processor import VoiceProcessor
        voice_processor = VoiceProcessor()
        
        # Process audio
        if enhance_audio:
            from src.utils.audio_processor import AudioProcessor
            audio_processor = AudioProcessor()
            audio_data = audio_processor.enhance_audio(audio_data)
        
        # Recognize speech
        recognized_text = voice_processor.recognize_speech(audio_data)
        
        if not recognized_text:
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Could not recognize speech",
                    "recognized_text": None,
                    "audio_used": "enhanced" if enhance_audio else "original",
                    "ai_response": None
                }
            )
        
        # Get AI response
        ai_response = voice_processor.get_ai_response(recognized_text)
        
        result = {
            "success": True,
            "recognized_text": recognized_text,
            "audio_used": "enhanced" if enhance_audio else "original",
            "ai_response": ai_response
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/analyze", tags=["Code Analysis"])
async def analyze_code(request: CodeRequest):
    """
    Analyze code using enhanced Gemini capabilities
    """
    try:
        # Get basic analysis
        analysis = code_helper.analyze_code(request.code, request.language)
        
        # Add security analysis if requested
        if request.analysis_type in ['full', 'security']:
            security_results = code_helper.check_security(request.code)
            analysis['security_analysis'] = security_results
        
        # Add best practices if requested
        if request.analysis_type in ['full', 'best_practices']:
            best_practices = code_helper.get_best_practices(request.code, request.language)
            analysis['best_practices'] = best_practices
            
        # Add suggestions
        suggestions = code_helper.generate_suggestions(request.code)
        
        # Calculate metrics directly
        # Split lines and filter out empty lines at the start and end
        lines = [line for line in request.code.splitlines() if line.strip()]
        
        metrics = {
            'total_lines': len(lines),
            'non_empty_lines': len(lines),
            'average_line_length': sum(len(line.strip()) for line in lines) / len(lines) if lines else 0
        }
        
        return {
            "success": True,
            "language": request.language,
            "analysis": analysis,
            "suggestions": suggestions,
            "complexity_score": analysis.get('complexity', 0),
            "security_issues": analysis.get('security_analysis', []),
            "best_practices": analysis.get('best_practices', []),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error analyzing code: {str(e)}")
        return {"success": False, "error": str(e)}

@app.post("/conversation", tags=["Conversation"])
async def manage_conversation(request: ConversationRequest):
    """
    Manage conversation state and history.
    Supports starting new conversations, continuing existing ones, and clearing history.
    """
    try:
        if request.action == "start":
            chat_id = voice_processor.create_new_chat()
            return {
                "success": True, 
                "message": "New conversation started",
                "chat_id": chat_id,
                "history": []
            }
            
        elif request.action == "continue" and request.text:
            response = voice_processor.get_ai_response(request.text, request.chat_id)
            history = voice_processor.get_history(request.chat_id)
            return {
                "success": True,
                "chat_id": voice_processor.active_chat_id,
                "response": response,
                "history": history
            }
            
        elif request.action == "clear":
            voice_processor.clear_history(request.chat_id)
            return {
                "success": True,
                "message": "Conversation history cleared",
                "chat_id": voice_processor.active_chat_id,
                "history": []
            }
            
        elif request.action == "list":
            chats = voice_processor.list_chats()
            return {
                "success": True,
                "chats": chats
            }
            
        elif request.action == "switch" and request.chat_id:
            chat = voice_processor.get_chat(request.chat_id)
            return {
                "success": True,
                "message": f"Switched to chat {request.chat_id}",
                "chat_id": request.chat_id,
                "history": voice_processor.get_history(request.chat_id)
            }
            
        else:
            raise HTTPException(status_code=400, detail="Invalid action or missing parameters")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts")
async def text_to_speech(
    text: str = Query(..., description="Text to convert to speech"),
    voice: Optional[str] = Query(None, description="Voice name to use"),
    rate: Optional[int] = Query(None, description="Speech rate (words per minute)"),
    volume: Optional[float] = Query(None, description="Volume level (0.0 to 1.0)")
):
    """Convert text to speech and return WAV audio"""
    try:
        # Convert text to speech
        audio_data = "Text to speech not implemented yet"
        
        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
            
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(22050)  # Sample rate
            wav_file.writeframes(audio_data)
            
        wav_buffer.seek(0)
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="speech.wav"'}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint for monitoring services.
    Returns basic system information and status.
    """
    return {
        "status": "healthy",
        "api_version": "1.0.0",
        "timestamp": time.time(),
        "system_info": {
            "python_version": platform.python_version(),
            "platform": platform.platform()
        }
    }

@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """WebSocket endpoint for real-time voice interaction"""
    try:
        await manager.connect(websocket)
        print("WebSocket connection opened")
        
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                
                # Parse JSON message
                try:
                    data = json.loads(message)
                    if 'type' not in data:
                        raise ValueError("Message type not specified")
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })
                    continue
                
                # Handle different message types
                if data['type'] == 'audio':
                    if 'audio' not in data:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Audio data not provided"
                        })
                        continue
                        
                    # Process audio
                    audio_data = base64.b64decode(data['audio'])
                    recognized_text = voice_processor.recognize_speech(audio_data)
                    
                    # Send recognition result
                    await websocket.send_json({
                        "type": "recognition",
                        "success": recognized_text is not None,
                        "text": recognized_text or "Could not recognize speech"
                    })
                    
                    if recognized_text:
                        # Get AI response with history
                        response = voice_processor.get_ai_response(recognized_text)
                        history = voice_processor.get_history()
                        
                        # Send text response with history
                        await websocket.send_json({
                            "type": "response",
                            "text": response,
                            "history": history
                        })
                        
                        # Generate and send audio response if needed
                        if data.get('return_audio', False):
                            audio_data = voice_processor.text_to_speech(response)
                            if audio_data:
                                await websocket.send_json({
                                    "type": "audio",
                                    "data": base64.b64encode(audio_data).decode()
                                })
                
                elif data['type'] == 'text':
                    if 'text' not in data:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Text not provided"
                        })
                        continue
                        
                    # Get AI response with history
                    response = voice_processor.get_ai_response(data['text'])
                    history = voice_processor.get_history()
                    
                    # Send text response with history
                    await websocket.send_json({
                        "type": "response",
                        "text": response,
                        "history": history
                    })
                    
                    # Generate and send audio response if needed
                    if data.get('return_audio', False):
                        audio_data = voice_processor.text_to_speech(response)
                        if audio_data:
                            await websocket.send_json({
                                "type": "audio",
                                "data": base64.b64encode(audio_data).decode()
                            })
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {data['type']}"
                    })
                    
            except WebSocketDisconnect:
                print("Client disconnected")
                await manager.disconnect(websocket)
                break
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()

@app.get("/voices", tags=["Speech"])
async def list_voices():
    """Get list of available TTS voices"""
    try:
        voices = "List of voices not implemented yet"
        return {"success": True, "voices": voices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", tags=["Info"])
async def root():
    """Get API information and status"""
    return {
        "name": "Jarvis AI Assistant API",
        "version": "1.0.0",
        "status": "operational",
        "features": {
            "voice_recognition": "Wit.ai powered speech recognition",
            "text_to_speech": "Customizable TTS with multiple voices",
            "code_analysis": "Multi-language code analysis and suggestions",
            "conversation": "Context-aware conversations with Gemini AI",
            "streaming": "Real-time voice interaction support"
        },
        "endpoints": {
            "/text": "Process text input with optional TTS",
            "/voice": "Process voice input with optional TTS",
            "/code/analyze": "Analyze and get suggestions for code",
            "/conversation": "Manage conversation state and history",
            "/tts": "Convert text to speech (streaming audio)",
            "/ws/voice": "WebSocket for real-time voice interaction",
            "/voices": "List available TTS voices"
        }
    }

# Main entry point
if __name__ == "__main__":
    # Start the ASGI server
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
