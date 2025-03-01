#!/usr/bin/env python
"""
Glitch setup helper script for Jarvis project.
This script helps configure the environment for Glitch deployment.
"""

import os
import sys
import subprocess
import json

def check_python_version():
    """Check and print the Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return version

def setup_glitch_environment():
    """Setup Glitch-specific configurations."""
    # Create package.json if it doesn't exist
    if not os.path.exists('package.json'):
        package_data = {
            "name": "jarvis-ai-assistant",
            "version": "1.0.0",
            "description": "Jarvis AI Assistant",
            "main": "api.py",
            "scripts": {
                "start": "python api.py"
            },
            "engines": {
                "node": "14.x"
            }
        }
        with open('package.json', 'w') as f:
            json.dump(package_data, f, indent=2)
        print("Created package.json with start script")

def install_dependencies():
    """Try different approaches to install dependencies."""
    print("Attempting to install dependencies...")
    
    # Approach 1: Try using the Glitch-specific requirements
    if os.path.exists('requirements-glitch.txt'):
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-glitch.txt'], 
                           check=True)
            print("Successfully installed dependencies from requirements-glitch.txt")
            return True
        except subprocess.CalledProcessError:
            print("Failed to install using requirements-glitch.txt")
    
    # Approach 2: Try installing core packages individually
    print("Trying to install core packages individually...")
    core_packages = [
        'google-generativeai',
        'fastapi',
        'uvicorn',
        'python-dotenv',
        'pydantic',
        'httpx',
        'requests'
    ]
    
    for package in core_packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=False)
            print(f"Installed {package}")
        except:
            print(f"Failed to install {package}")
    
    return True

def main():
    """Main entry point for the script."""
    print("Starting Glitch setup for Jarvis...")
    
    version = check_python_version()
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Warning: Python 3.8+ is recommended for this project")
    
    setup_glitch_environment()
    install_dependencies()
    
    print("\nSetup complete. If you're still experiencing issues:")
    print("1. Check Glitch logs for detailed error messages")
    print("2. Make sure your environment variables are set in the Glitch project settings")
    print("3. Try running 'python api.py' manually to see specific errors")

if __name__ == "__main__":
    main()
