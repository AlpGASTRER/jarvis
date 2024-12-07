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
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
import json
import asyncio
import time
import io
import wave
import base64
from src.utils.enhanced_code_helper import EnhancedCodeHelper
import os

# Initialize core components
code_helper = EnhancedCodeHelper()

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
@app.post("/text", response_model=TextResponse, tags=["Text"])
async def process_text(request: TextRequest):
    """
    Process text input and get AI response.
    Supports general queries, code-related questions, and voice commands.
    Can return both text and audio responses.
    """
    try:
        # Initialize Gemini AI
        import google.generativeai as genai
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel('gemini-pro')
        
        # Process based on mode
        if request.mode == "code":
            # Use code helper for code-related queries
            response = code_helper.get_code_help(request.text)
        else:
            # Get general AI response
            response = model.generate_content(request.text)
            response = response.text
            
        result = {
            "success": True,
            "response": response,
            "audio_response": None
        }
        
        # Generate audio response if requested
        if request.return_audio:
            from src.utils.voice_processor import VoiceProcessor
            processor = VoiceProcessor()
            audio_data = processor.text_to_speech(
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
        # Get code analysis
        analysis = code_helper.analyze_code(request.code)
        if not analysis:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "language": request.language or "unknown",
                    "analysis": {},
                    "suggestions": [],
                    "error": "Code analysis failed. Please check the code and try again."
                }
            )

        # Get additional analysis based on type
        suggestions = code_helper.get_suggestions(request.code, analysis.language)
        best_practices = None
        security_issues = None
        
        if request.analysis_type in ['full', 'best_practices']:
            best_practices = code_helper.get_best_practices(request.code, analysis.language)
            
        if request.analysis_type in ['full', 'security']:
            security_issues = code_helper.analyze_security(request.code, analysis.language)

        # Convert metrics to response format
        metrics = None
        if analysis.metrics:
            metrics = {
                "total_lines": analysis.metrics.get('total_lines', 0),
                "non_empty_lines": analysis.metrics.get('non_empty_lines', 0),
                "average_line_length": analysis.metrics.get('average_line_length', 0.0),
                "num_functions": analysis.metrics.get('num_functions'),
                "num_classes": analysis.metrics.get('num_classes'),
                "num_imports": analysis.metrics.get('num_imports'),
                "cyclomatic_complexity": analysis.complexity
            }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "language": analysis.language,
                "analysis": {
                    "code_blocks": analysis.code_blocks,
                    "references": analysis.references
                },
                "suggestions": suggestions or [],
                "best_practices": best_practices,
                "security_issues": security_issues,
                "complexity_score": analysis.complexity,
                "metrics": metrics
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "language": request.language or "unknown",
                "analysis": {},
                "suggestions": [],
                "error": str(e)
            }
        )

@app.post("/conversation", tags=["Conversation"])
async def manage_conversation(request: ConversationRequest):
    """
    Manage conversation state and history.
    Supports starting new conversations, continuing existing ones, and clearing history.
    """
    try:
        if request.action == "start":
            return {"success": True, "conversation_id": "Conversation ID not implemented yet"}
            
        elif request.action == "continue" and request.text:
            if not request.conversation_id:
                raise HTTPException(status_code=400, detail="Conversation ID required")
            return {"success": True, "response": "Continuing conversation not implemented yet"}
            
        elif request.action == "clear":
            if request.conversation_id:
                return {"success": True, "message": "Clearing conversation not implemented yet"}
            else:
                return {"success": True, "message": "Clearing all conversations not implemented yet"}
            
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

@app.websocket("/ws/voice")
async def websocket_voice(websocket: WebSocket):
    """WebSocket endpoint for real-time voice interaction"""
    try:
        await websocket.accept()
        
        # Initialize processors
        from src.utils.voice_processor import VoiceProcessor
        voice_processor = VoiceProcessor()
        
        while True:
            try:
                # Receive audio data
                data = await websocket.receive_json()
                audio_data = base64.b64decode(data['audio'])
                
                # Process audio
                recognized_text = voice_processor.recognize_speech(audio_data)
                
                # Send recognition result
                await websocket.send_json({
                    "type": "text",
                    "recognized": recognized_text or "Could not recognize speech",
                    "response": None
                })
                
                if recognized_text:
                    # Get AI response
                    response = voice_processor.get_ai_response(recognized_text)
                    
                    # Send text response
                    await websocket.send_json({
                        "type": "text",
                        "recognized": recognized_text,
                        "response": response
                    })
                    
                    # Generate and send audio response
                    audio_data = voice_processor.text_to_speech(response)
                    if audio_data:
                        await websocket.send_json({
                            "type": "audio",
                            "data": base64.b64encode(audio_data).decode()
                        })
                        
            except WebSocketDisconnect:
                print("Client disconnected")
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
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
