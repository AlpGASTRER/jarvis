# Jarvis AI Assistant

A powerful AI assistant that combines voice interaction, code analysis, and natural language processing using Google's Gemini API.

## Features

### Voice Interaction
- Real-time voice recognition
- Text-to-speech with customizable voices
- Audio enhancement and noise reduction
- Support for both streaming and file-based audio

### Code Analysis
- Multi-language support (12+ programming languages)
- Syntax analysis and validation
- Code improvement suggestions
- Best practices recommendations
- Security vulnerability detection
- Complexity scoring
- Language-specific analysis

### AI Capabilities
- Powered by Google's Gemini AI
- Context-aware conversations
- Code understanding and generation
- Natural language processing
- Memory of recent interactions

## Supported Programming Languages
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

## Getting Started

### Prerequisites
- Python 3.9 or higher
- API Keys:
  - Google API Key (for Gemini)
  - Wit.ai API Key (for voice recognition)

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd jarvis
```

2. Create a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
echo WIT_EN_KEY=your_wit_key > .env
echo GOOGLE_API_KEY=your_google_key >> .env
```

5. Run the API:
```bash
python api.py
```

## API Documentation

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for detailed API documentation.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for deployment instructions.

## Development

### Project Structure
```
jarvis/
├── api.py              # FastAPI application
├── jarvis.py           # Core Jarvis class
├── requirements.txt    # Python dependencies
├── src/
│   └── utils/
│       ├── enhanced_code_helper.py    # Code analysis
│       ├── audio_processor.py         # Audio processing
│       ├── voice_processor.py         # Voice recognition
│       └── tts_processor.py           # Text-to-speech
├── tests/             # Test files
└── docs/              # Documentation
```

### Testing
Run the test client:
```bash
python test_voice_api.py
```

## Security

- API key authentication
- Rate limiting
- Input validation
- Error handling
- Security analysis for code
- CORS protection

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini AI for natural language processing
- FastAPI for the web framework
- Wit.ai for voice recognition
