# Jarvis AI Assistant API Documentation

## Overview

The Jarvis AI Assistant API provides a comprehensive set of endpoints for voice recognition, text-to-speech, code analysis, and natural language processing. Built with FastAPI and powered by Google's Gemini AI, it offers both REST and WebSocket interfaces for real-time interaction.

## Base URL

```
http://localhost:8000
```

## Authentication

API key authentication is required for all endpoints except documentation:
```http
X-API-Key: your-api-key
```

Environment variables required:
- `WIT_EN_KEY`: Wit.ai API key for voice recognition
- `GOOGLE_API_KEY`: Google API key for Gemini AI

## Rate Limiting

- Default: 100 requests per minute per IP
- WebSocket: 10 concurrent connections per IP
- Voice endpoints: 50 requests per minute per IP

## Endpoints

### Code Analysis

#### POST `/code/analyze`
Analyze code and provide suggestions.

**Request Body:**
```json
{
    "code": "string",
    "language": "python | javascript | typescript | java | cpp | csharp | go | rust | ruby | php | swift | kotlin",
    "analysis_type": "full | syntax | suggestions | best_practices | security"
}
```

**Response:**
```json
{
    "success": true,
    "language": "string",
    "analysis": {
        "syntax": "string",
        "suggestions": ["string"],
        "best_practices": ["string"],
        "security": ["string"]
    },
    "suggestions": ["string"],
    "best_practices": ["string"],
    "security_issues": ["string"],
    "complexity_score": 0.0
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

## Error Handling

All endpoints return standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 429: Too Many Requests
- 500: Internal Server Error

Error response format:
```json
{
    "success": false,
    "error": "string",
    "detail": "string"
}
```

## Security

- All endpoints require API key authentication
- Rate limiting per IP address
- Input validation and sanitization
- CORS protection
- No sensitive information in error messages
- Security analysis for code endpoints

## Best Practices

1. Always set appropriate headers:
   - `Content-Type: application/json`
   - `X-API-Key: your-api-key`

2. Handle rate limiting with exponential backoff

3. Implement proper error handling

4. Use WebSocket for real-time voice interaction

5. Consider caching frequently requested data

## Examples

See the `test_voice_api.py` file for complete examples of using all endpoints.
