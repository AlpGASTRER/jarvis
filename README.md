# Jarvis AI Assistant

A powerful AI assistant that combines voice interaction, code analysis, and natural language processing capabilities.

## Features

- **Voice Interaction**
  - Real-time voice recognition
  - Fast local text-to-speech using pyttsx3
  - WebSocket-based audio streaming
  - Response caching for improved performance

- **Multi-Chat Support**
  - Multiple concurrent chat sessions
  - Persistent chat history
  - Context-aware conversations
  - Session management (create, switch, clear)

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

## Chat API Usage

The `/conversation` endpoint supports multiple chat actions:

```python
# Create a new chat
POST /conversation
{
    "action": "new"
}

# Continue a conversation
POST /conversation
{
    "action": "continue",
    "chat_id": "your-chat-id",  # Optional, uses active chat if not provided
    "text": "Your message here"
}

# Switch between chats
POST /conversation
{
    "action": "switch",
    "chat_id": "target-chat-id"
}

# Clear chat history
POST /conversation
{
    "action": "clear",
    "chat_id": "chat-id-to-clear"  # Optional, clears active chat if not provided
}

# List all active chats
POST /conversation
{
    "action": "list"
}
```

Each chat maintains its own conversation history, allowing for context-aware responses. When a chat becomes too long and hits the token limit, simply start a new chat or clear the current one.

## Architecture

- FastAPI backend with WebSocket support
- Local TTS engine for fast responses
- Caching system for frequently used responses
- Multi-chat session management
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

## Roadmap

### Enhanced AI Capabilities
- Conversation memory and context awareness
- Specialized assistant modes (coding, teaching, etc.)
- Multi-model support (Gemini, Claude, GPT-4)
- Improved code understanding and generation

### Advanced Code Features
- Live code execution environment
- Git integration for repository analysis
- Multi-file project analysis
- Code generation with templates
- Auto-completion suggestions
- Dependency analysis
- Code refactoring suggestions

### Voice Enhancements
- Multiple voice options and personalities
- Voice emotion detection
- Advanced background noise cancellation
- Speaker identification
- Custom wake word ("Hey Jarvis")
- Multi-language support

### System Integration
- Desktop application integration
- IDE plugins (VS Code, PyCharm)
- System commands execution
- File system operations
- Browser integration
- Cross-platform support

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
