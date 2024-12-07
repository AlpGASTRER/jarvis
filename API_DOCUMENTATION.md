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
- `GOOGLE_API_KEY`: Google API key for Gemini AI (Required)
- `WIT_EN_KEY`: Wit.ai API key for voice recognition (Optional, for voice features)

## Rate Limiting

- Default: 100 requests per minute per IP
- WebSocket: 10 concurrent connections per IP
- Voice endpoints: 50 requests per minute per IP

## Endpoints

### Code Analysis

#### POST `/code/analyze`
Analyze code and provide suggestions, best practices, and security analysis.

**Features:**
- Automatic language detection
- Multiple analysis types
- Complexity metrics
- Security vulnerability scanning
- Best practices recommendations

**Request Body:**
```json
{
    "code": "string",
    "language": "python | javascript | typescript | java | cpp | csharp | go | rust | ruby | php | swift | kotlin | null",
    "analysis_type": "full | syntax | suggestions | best_practices | security",
    "config_path": "string | null"
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/code/analyze" \
     -H "Content-Type: application/json" \
     -d '{
         "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
         "analysis_type": "full"
     }'
```

**Example Response:**
```json
{
    "success": true,
    "language": "python",
    "analysis": {
        "code_blocks": [
            "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        ],
        "references": []
    },
    "suggestions": [
        "Add input validation for negative numbers",
        "Consider using iteration instead of recursion for better performance",
        "Add type hints for better code documentation"
    ],
    "best_practices": [
        "Follow PEP 8 style guide",
        "Add docstring to document function purpose and parameters",
        "Consider adding error handling"
    ],
    "security_issues": [
        "Add input validation to prevent stack overflow",
        "Consider adding maximum recursion depth"
    ],
    "complexity_score": 3,
    "metrics": {
        "total_lines": 4,
        "non_empty_lines": 4,
        "average_line_length": 15.5,
        "num_functions": 1,
        "num_classes": 0,
        "num_imports": 0,
        "cyclomatic_complexity": 2
    }
}
```

### Voice Processing

#### POST `/voice`
Process voice input and get AI response with optional text-to-speech.

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

**Example using curl with a WAV file:**
```bash
curl -X POST "http://localhost:8000/voice" \
     -H "Content-Type: multipart/form-data" \
     -F "audio_file=@recording.wav" \
     -F "enhance_audio=true"
```

### Text Processing

#### POST `/text`
Process text input and get AI response with optional code analysis.

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

**Example Request:**
```bash
curl -X POST "http://localhost:8000/text" \
     -H "Content-Type: application/json" \
     -d '{
         "text": "What is the time complexity of quicksort?",
         "mode": "code",
         "return_audio": true
     }'
```

### Text-to-Speech

#### GET `/tts`
Convert text to speech with customizable voice settings.

**Query Parameters:**
- `text` (required): Text to convert
- `voice`: Voice name (default: system default)
- `rate`: Speech rate in WPM (default: 175)
- `volume`: Volume level 0.0-1.0 (default: 1.0)

**Example Request:**
```bash
curl "http://localhost:8000/tts?text=Hello%20World&voice=en-US-1&rate=200" \
     --output speech.wav
```

### WebSocket

#### WS `/ws/voice`
Real-time voice interaction with streaming responses.

**Connection Example (JavaScript):**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/voice?return_audio=true');

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    if (response.type === 'text') {
        console.log('Recognized:', response.recognized);
        console.log('Response:', response.response);
    } else if (response.type === 'audio') {
        // Handle audio response
        const audio = new Audio(`data:audio/wav;base64,${response.data}`);
        audio.play();
    }
};

// Send audio data
ws.send(JSON.stringify({
    audio: base64AudioData
}));
```

## Error Handling

All endpoints return standard HTTP status codes with detailed error messages:

```json
{
    "success": false,
    "error": "Detailed error message",
    "error_code": "ERROR_CODE",
    "details": {
        "field": "Additional error details"
    }
}
```

Common status codes:
- 200: Success
- 400: Bad Request (invalid input)
- 401: Unauthorized (missing/invalid API key)
- 403: Forbidden (rate limit exceeded)
- 422: Validation Error (invalid request body)
- 500: Internal Server Error

## Best Practices

1. **Rate Limiting:**
   - Implement client-side rate limiting
   - Use exponential backoff for retries

2. **Error Handling:**
   - Always check the `success` field in responses
   - Handle network errors gracefully
   - Implement proper retry logic

3. **WebSocket Usage:**
   - Implement reconnection logic
   - Handle connection timeouts
   - Process messages sequentially

4. **Audio Processing:**
   - Use recommended audio formats (WAV, 16-bit PCM)
   - Follow sample rate guidelines (16kHz for voice)
   - Keep audio chunks under 15 seconds

## Support

For issues and feature requests, please contact:
- Email: support@jarvis-ai.com
- GitHub: https://github.com/jarvis-ai/issues
