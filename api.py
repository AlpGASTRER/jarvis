import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, Query, Depends, Header, WebSocketDisconnect, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import json
import asyncio
import time
from jarvis import Jarvis
import io
import wave
import base64

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Jarvis
jarvis = Jarvis()

# Request Models
class TextRequest(BaseModel):
    text: str = Field(..., description="The text input to process")
    mode: Optional[str] = Field("general", description="Processing mode: 'general', 'code', or 'voice'")
    return_audio: Optional[bool] = Field(False, description="Whether to return audio response")
    voice_settings: Optional[Dict[str, Any]] = Field(None, description="Custom voice settings (rate, volume, voice)")

class AudioRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    channels: int = Field(1, description="Number of audio channels")
    return_audio: Optional[bool] = Field(False, description="Whether to return audio response")
    enhance_audio: Optional[bool] = Field(True, description="Whether to apply noise reduction")

class CodeRequest(BaseModel):
    code: str = Field(..., description="Code to analyze")
    language: Optional[str] = Field(None, description="Programming language")
    analysis_type: Optional[str] = Field("full", description="Type of analysis: 'full', 'syntax', 'suggestions'")

class ConversationRequest(BaseModel):
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

@app.post("/code/analyze", tags=["Code Analysis"])
async def analyze_code(request: CodeRequest):
    """
    Analyze code and provide suggestions.
    Supports multiple programming languages and different types of analysis.
    """
    try:
        # Auto-detect language if not specified
        if not request.language:
            request.language = jarvis.code_helper.detect_language(request.code)
            
        # Get analysis based on type
        if request.analysis_type == "syntax":
            analysis = jarvis.code_helper.analyze_syntax(request.code, request.language)
        elif request.analysis_type == "suggestions":
            analysis = jarvis.code_helper.get_suggestions(request.code, request.language)
        else:
            analysis = jarvis.code_helper.analyze_code(request.code, request.language)
            
        if analysis:
            return {
                "success": True,
                "language": request.language,
                "analysis": analysis
            }
        else:
            raise HTTPException(status_code=400, detail="Could not analyze code")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
