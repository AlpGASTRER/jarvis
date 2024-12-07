# Jarvis - AI Voice Assistant

A voice-activated AI assistant powered by Google's Gemini API, capable of handling both programming queries and general conversation.

## Features

- Voice-activated interaction
- Natural language processing using Google's Gemini AI
- Programming assistance and code help
- General conversation capabilities
- Text-to-Speech response system

## Prerequisites

- Python 3.9+
- Conda (recommended for environment management)
- Google API key for Gemini
- Wit.ai API key for voice recognition

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jarvis.git
cd jarvis
```

2. Create and activate a conda environment:
```bash
conda create -n jarvis python=3.9
conda activate jarvis
```

3. Install required packages:
```bash
python -m pip install google-generativeai==0.3.1 grpcio==1.59.3
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
WIT_AI_KEY=your_wit_ai_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

1. Activate the conda environment:
```bash
conda activate jarvis
```

2. Run the assistant:
```bash
python jarvis.py
```

3. Choose your preferred mode:
   - Voice mode (1): Interact with Jarvis using voice commands
   - Chat mode (2): Interact with Jarvis using text input

## Features

### Voice Commands
- Natural language processing for voice input
- Clear voice output using text-to-speech
- Automatic speech recognition with noise reduction

### AI Capabilities
- Programming assistance and code explanations
- General knowledge questions
- Conversational interactions
- Context-aware responses

## Project Structure

```
jarvis/
├── src/
│   └── utils/
│       └── code_helper.py
├── jarvis.py
├── requirements.txt
└── .env
```

## Dependencies

- `google-generativeai`: Google's Gemini AI API
- `speech_recognition`: Voice recognition
- `pyttsx3`: Text-to-speech conversion
- `python-dotenv`: Environment variable management
- Additional dependencies listed in `requirements.txt`

## Deployment

For detailed deployment instructions, including:
- Local development setup
- Docker deployment
- Cloud deployment (AWS)
- Kubernetes deployment
- Monitoring and maintenance
- Security considerations
- Troubleshooting guide

Please refer to our comprehensive [Deployment Guide](DEPLOYMENT.md).

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
