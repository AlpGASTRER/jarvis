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

Dependencies:
- FastAPI: Web framework
- Pydantic: Data validation
- uvicorn: ASGI server
- Jarvis: Core AI assistant functionality
"""

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, Query, Depends, Header, WebSocketDisconnect, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import json
import asyncio
import time
from jarvis import Jarvis
import io
import wave
import base64

# Initialize FastAPI with metadata
app = FastAPI(
    title="Jarvis AI Assistant API",
    description="""
    Comprehensive API for Jarvis AI Assistant featuring:
    - Voice Recognition with Wit.ai
    - Text-to-Speech with customizable voices
    - Code Analysis and Suggestions
    - Natural Language Processing with Google's Gemini
    - Real-time Conversation
    """,
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

# Initialize global Jarvis instance
jarvis = Jarvis()

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
    """
    Model for code analysis requests.
    
    Attributes:
        code: Source code to analyze
        language: Programming language
        analysis_type: Type of analysis to perform
        config_path: Optional path to custom analysis configuration
    """
    code: str = Field(..., description="Code to analyze")
    language: Optional[str] = Field(None, description="Programming language")
    analysis_type: Optional[str] = Field("full", description="Type of analysis")
    config_path: Optional[str] = Field(None, description="Path to custom analysis configuration")

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
    metrics: Optional[CodeMetrics]
    error: Optional[str] = None

class ConversationRequest(BaseModel):
    """
    Model for conversation management requests.
    
    Attributes:
        action: Action to perform (start/continue/clear)
        text: Input text for continuation
        conversation_id: ID of conversation to continue
    """
    action: str = Field(..., description="Action to perform: 'start', 'continue', 'clear'")
    text: Optional[str] = Field(None, description="Text for continuation")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continuation")

# API Routes
@app.post("/text", tags=["Text Processing"])
async def process_text(request: TextRequest):
    """
    Process text input and get AI response.
    Supports general queries, code-related questions, and voice commands.
    Can return both text and audio responses.
    """
    try:
        # Process based on mode
        if request.mode == "code":
            response = jarvis.handle_code_query(request.text)
        else:
            response = jarvis.process_command(request.text)
        
        result = {"success": True, "response": response}
        
        # Generate audio if requested
        if request.return_audio:
            # Apply custom voice settings if provided
            if request.voice_settings:
                jarvis.tts_processor.update_settings(request.voice_settings)
                
            tts_result = jarvis.tts_processor.text_to_speech_base64(response)
            if tts_result["success"]:
                result["audio_response"] = tts_result
            else:
                result["audio_error"] = tts_result["error"]
                
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice")
async def process_voice(
    audio_file: UploadFile = File(..., description="Audio file to process"),
    enhance_audio: bool = Form(False, description="Whether to apply audio enhancement")
):
    """Process voice input and return AI response"""
    try:
        # Read audio data
        audio_data = await audio_file.read()
        
        # Process voice using Jarvis
        jarvis = Jarvis()
        result = jarvis.process_voice(audio_data, enhance_audio=enhance_audio)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        # Get AI response
        text = result["text"]
        ai_response = jarvis.process_text(text)
        
        if not isinstance(ai_response, dict):
            ai_response = {"success": True, "response": ai_response}
            
        if not ai_response["success"]:
            raise HTTPException(status_code=500, detail=ai_response["error"])
            
        return {
            "recognized_text": text,
            "audio_used": result["audio_used"],
            "ai_response": ai_response["response"]
        }
        
    except Exception as e:
        print(f"Error in process_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/analyze", response_model=CodeResponse, tags=["Code Analysis"])
async def analyze_code(request: CodeRequest):
    """
    Analyze code and provide suggestions.
    
    Supports multiple programming languages and different types of analysis.
    Enhanced with detailed metrics, security analysis, and best practices.
    
    Available languages:
    - Python: General-purpose, AI/ML, Web
    - JavaScript/TypeScript: Web, Node.js
    - Java: Enterprise, Android
    - C++: Systems, Games
    - C#: Windows, Unity
    - Go: Cloud, Systems
    - Rust: Systems, WebAssembly
    - Ruby: Web, Scripting
    - PHP: Web Development
    - Swift: iOS, macOS
    - Kotlin: Android, JVM
    
    Returns:
        CodeResponse: Comprehensive analysis results
        
    Raises:
        HTTPException: If analysis fails or language is not supported
    """
    try:
        # Initialize code helper with custom config if provided
        code_helper = jarvis.code_helper
        if request.config_path:
            code_helper = jarvis.initialize_code_helper(request.config_path)
        
        # Perform code analysis
        analysis = code_helper.analyze_code(request.code)
        if not analysis:
            raise HTTPException(
                status_code=500,
                detail="Code analysis failed. Please check the code and try again."
            )
        
        # Get additional analysis based on type
        suggestions = code_helper.get_suggestions(request.code, analysis.language)
        best_practices = (
            code_helper.get_best_practices(request.code, analysis.language)
            if request.analysis_type in ['full', 'best_practices']
            else None
        )
        security_issues = (
            code_helper.analyze_security(request.code, analysis.language)
            if request.analysis_type in ['full', 'security']
            else None
        )
        
        # Convert metrics to CodeMetrics model
        metrics = None
        if analysis.metrics:
            metrics = CodeMetrics(
                total_lines=analysis.metrics.get('total_lines', 0),
                non_empty_lines=analysis.metrics.get('non_empty_lines', 0),
                average_line_length=analysis.metrics.get('average_line_length', 0.0),
                num_functions=analysis.metrics.get('num_functions'),
                num_classes=analysis.metrics.get('num_classes'),
                num_imports=analysis.metrics.get('num_imports'),
                cyclomatic_complexity=analysis.complexity
            )
        
        return CodeResponse(
            success=True,
            language=analysis.language,
            analysis={'code_blocks': analysis.code_blocks, 'references': analysis.references},
            suggestions=suggestions,
            best_practices=best_practices,
            security_issues=security_issues,
            complexity_score=analysis.complexity,
            metrics=metrics
        )
        
    except Exception as e:
        return CodeResponse(
            success=False,
            language=request.language or 'unknown',
            analysis={},
            suggestions=[],
            error=str(e)
        )

@app.post("/conversation", tags=["Conversation"])
async def manage_conversation(request: ConversationRequest):
    """
    Manage conversation state and history.
    Supports starting new conversations, continuing existing ones, and clearing history.
    """
    try:
        if request.action == "start":
            conversation_id = jarvis.start_new_conversation()
            return {"success": True, "conversation_id": conversation_id}
            
        elif request.action == "continue" and request.text:
            if not request.conversation_id:
                raise HTTPException(status_code=400, detail="Conversation ID required")
            response = jarvis.continue_conversation(request.conversation_id, request.text)
            return {"success": True, "response": response}
            
        elif request.action == "clear":
            if request.conversation_id:
                jarvis.clear_conversation(request.conversation_id)
            else:
                jarvis.clear_all_conversations()
            return {"success": True, "message": "Conversation(s) cleared"}
            
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
        # Initialize Jarvis
        jarvis = Jarvis()
        
        # Convert text to speech
        audio_data = jarvis.text_to_speech(text, voice, rate, volume)
        
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

@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """WebSocket endpoint for real-time voice interaction"""
    try:
        await websocket.accept()
        
        while True:
            try:
                # Receive audio data
                audio_data = await websocket.receive()
                
                # Extract bytes from the message
                if audio_data["type"] == "websocket.receive":
                    if "bytes" in audio_data:
                        audio_bytes = audio_data["bytes"]
                    elif "text" in audio_data:
                        # Handle base64 encoded data
                        audio_bytes = base64.b64decode(audio_data["text"])
                    else:
                        raise ValueError("No audio data received")
                else:
                    continue
                
                # Process voice using Jarvis
                jarvis = Jarvis()
                result = jarvis.process_voice(audio_bytes)
                
                if result["success"]:
                    # Get AI response
                    text = result["text"]
                    ai_response = jarvis.process_text(text)
                    
                    if not isinstance(ai_response, dict):
                        ai_response = {"success": True, "response": ai_response}
                    
                    if ai_response["success"]:
                        response_data = {
                            "text": text,
                            "response": ai_response["response"]
                        }
                    else:
                        response_data = {"error": ai_response["error"]}
                else:
                    response_data = {"error": result["error"]}
                    
                await websocket.send_json(response_data)
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket error: {str(e)}")
                await websocket.send_json({"error": str(e)})
                break
                
    except Exception as e:
        print(f"WebSocket connection error: {str(e)}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.close(code=1001)

@app.get("/voices", tags=["Speech"])
async def list_voices():
    """Get list of available TTS voices"""
    try:
        voices = jarvis.tts_processor.list_voices()
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
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
