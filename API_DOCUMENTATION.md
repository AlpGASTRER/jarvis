# Jarvis AI Assistant API Documentation

## Overview

The Jarvis AI Assistant API provides a comprehensive set of endpoints for voice recognition, text-to-speech, code analysis, and natural language processing. Built with FastAPI and powered by Google's Gemini AI, it offers both REST and WebSocket interfaces for real-time interaction.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently uses API keys through environment variables:
- `WIT_EN_KEY`: Wit.ai API key for voice recognition
- `GOOGLE_API_KEY`: Google API key for Gemini AI

## Endpoints

### Text Processing

#### POST `/text`
Process text input and get AI response.

**Request Body:**
```json
{
    "text": "string",
    "mode": "general | code | voice",
    "return_audio": false,
    "voice_settings": {
        "rate": 175,
        "volume": 1.0,
        "voice": "string"
    }
}
```

**Response:**
```json
{
    "success": true,
    "response": "string",
    "audio_response": {
        "audio_base64": "string",
        "sample_rate": 22050,
        "channels": 1
    }
}
```

### Voice Processing

#### POST `/voice`
Process voice input and get AI response.

**Request Body:**
```json
{
    "audio_base64": "string",
    "sample_rate": 16000,
    "channels": 1,
    "return_audio": false,
    "enhance_audio": true
}
```

**Response:**
```json
{
    "success": true,
    "recognized_text": "string",
    "response": "string",
    "audio_used": "original | enhanced",
    "audio_response": {
        "audio_base64": "string",
        "sample_rate": 22050,
        "channels": 1
    }
}
```

### Code Analysis

#### POST `/code/analyze`
Analyze code and get suggestions.

**Request Body:**
```json
{
    "code": "string",
    "language": "python | javascript | java",
    "analysis_type": "full | syntax | suggestions"
}
```

**Response:**
```json
{
    "success": true,
    "language": "string",
    "analysis": {
        "complexity": "number",
        "suggestions": ["string"],
        "code_blocks": ["string"],
        "references": ["string"]
    }
}
```

### Conversation Management

#### POST `/conversation`
Manage conversation state and history.

**Request Body:**
```json
{
    "action": "start | continue | clear",
    "text": "string",
    "conversation_id": "string"
}
```

**Response:**
```json
{
    "success": true,
    "conversation_id": "string",
    "response": "string"
}
```

### Text-to-Speech

#### GET `/tts`
Convert text to speech and stream audio.

**Query Parameters:**
- `text` (required): Text to convert
- `voice`: Voice name to use
- `rate`: Speech rate (WPM)
- `volume`: Volume level (0.0-1.0)

**Response:**
- Content-Type: audio/wav
- Streaming audio data

#### GET `/voices`
List available TTS voices.

**Response:**
```json
{
    "success": true,
    "voices": [
        {
            "name": "string",
            "id": "string",
            "gender": "string"
        }
    ]
}
```

### WebSocket

#### WS `/ws/voice`
Real-time voice interaction.

**Query Parameters:**
- `return_audio`: Whether to return audio responses (default: true)
- `sample_rate`: Audio sample rate in Hz (default: 16000)
- `channels`: Number of audio channels (default: 1)
- `enhance_audio`: Whether to apply noise reduction (default: true)

**WebSocket Messages:**

Client to Server:
```json
{
    "audio": "base64_string"
}
```

Server to Client:
```json
{
    "type": "text",
    "recognized": "string",
    "response": "string"
}
```
or
```json
{
    "type": "audio",
    "data": "base64_string"
}
```
or
```json
{
    "type": "error",
    "error": "string"
}
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- 200: Success
- 400: Bad Request
- 500: Internal Server Error

Error responses include a detail message:
```json
{
    "detail": "Error message"
}
```

## Rate Limiting

Currently no rate limiting implemented. Consider adding in production.

## CORS

CORS is enabled for all origins (*). Update the settings in production.

## Examples

### Python Example
```python
import requests
import base64
import json

# Text processing
response = requests.post(
    "http://localhost:8000/text",
    json={
        "text": "What is Python?",
        "return_audio": True
    }
)

# Voice processing
with open("audio.wav", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()
    
response = requests.post(
    "http://localhost:8000/voice",
    json={
        "audio_base64": audio_base64,
        "return_audio": True
    }
)

# Code analysis
response = requests.post(
    "http://localhost:8000/code/analyze",
    json={
        "code": "def hello(): print('Hello')",
        "language": "python"
    }
)
```

### JavaScript Example
```javascript
// WebSocket voice interaction
const ws = new WebSocket('ws://localhost:8000/ws/voice');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'text') {
        console.log('Recognized:', data.recognized);
        console.log('Response:', data.response);
    } else if (data.type === 'audio') {
        // Play audio
        const audio = new Audio(`data:audio/wav;base64,${data.data}`);
        audio.play();
    }
};

// Send audio data
ws.send(JSON.stringify({
    audio: base64AudioData
}));
```

## Testing

Use the provided test client:
```bash
python test_voice_api.py
```

The test client supports:
1. Testing voice recognition with TTS response
2. Testing real-time conversation with streaming
3. Testing direct text-to-speech conversion
4. Testing code analysis
