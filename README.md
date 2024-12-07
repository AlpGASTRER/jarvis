# Jarvis AI Assistant

A powerful AI assistant that combines voice interaction, code analysis, and natural language processing capabilities.

## Features

- **Voice Interaction**
  - Real-time voice recognition
  - Fast local text-to-speech using pyttsx3
  - WebSocket-based audio streaming
  - Response caching for improved performance

- **Code Analysis**
  - Semantic code understanding
  - Best practices recommendations
  - Security vulnerability detection
  - Performance optimization suggestions
  - Code complexity metrics

- **AI Integration**
  - Google's Gemini AI for natural language processing
  - Pre-warmed model for faster responses
  - Context-aware conversations
  - Code-aware responses

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
GOOGLE_API_KEY=your_api_key_here
```

3. Run the server:
```bash
uvicorn api:app --reload
```

4. Connect to:
- WebSocket: `ws://localhost:8000/ws`
- HTTP API: `http://localhost:8000`

## Architecture

- FastAPI backend with WebSocket support
- Local TTS engine for fast responses
- Caching system for frequently used responses
- Modular design with separate processors for voice, code, and AI

## Performance Optimizations

- Local TTS using pyttsx3 instead of gTTS
- Response and audio caching
- Pre-warmed AI model
- Efficient WebSocket connection management
- Optimized file handling

## Security

- Input validation for file uploads
- Secure WebSocket connections
- Environment variable based configuration
- API key protection

## Contributing

Feel free to open issues or submit pull requests for:
- New features
- Bug fixes
- Documentation improvements
- Performance optimizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini AI for natural language processing
- FastAPI for the web framework
- Wit.ai for voice recognition
