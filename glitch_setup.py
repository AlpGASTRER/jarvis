#!/usr/bin/env python
"""
Glitch setup helper script for Jarvis project.
This script helps configure the environment for Glitch deployment.
Compatible with Python 2.7 and Python 3.x
"""

import os
import sys
import subprocess
import json

def check_python_version():
    """Check and print the Python version."""
    version = sys.version_info
    print("Python version: {0}.{1}.{2}".format(version.major, version.minor, version.micro))
    return version

def setup_glitch_environment():
    """Setup Glitch-specific configurations."""
    # Create .python-version file if it doesn't exist
    if not os.path.exists('.python-version'):
        with open('.python-version', 'w') as f:
            f.write('3.8.10')
        print("Created .python-version file")
        
    # Create package.json if it doesn't exist
    if not os.path.exists('package.json'):
        package_data = {
            "name": "jarvis-ai-assistant",
            "version": "1.0.0",
            "description": "Jarvis AI Assistant",
            "main": "api.py",
            "scripts": {
                "start": "python3 api.py"
            },
            "engines": {
                "node": "14.x"
            }
        }
        with open('package.json', 'w') as f:
            json.dump(package_data, f, indent=2)
        print("Created package.json with start script")

def install_python3():
    """Attempt to ensure Python 3 is available."""
    print("Checking for Python 3...")
    try:
        # Try to run python3 --version
        process = subprocess.Popen(["python3", "--version"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output = stdout.decode() if stdout else stderr.decode()
        print("Python3 check output: {0}".format(output))
        
        if process.returncode != 0:
            print("Python3 command failed, Python 3 may not be installed.")
            return False
            
        return True
    except Exception as e:
        print("Error checking Python 3: {0}".format(str(e)))
        return False

def install_dependencies():
    """Try different approaches to install dependencies."""
    print("Attempting to install dependencies...")
    
    python_cmd = "python3" if install_python3() else sys.executable
    
    # Approach 1: Try using the Glitch-specific requirements
    if os.path.exists('requirements-glitch.txt'):
        try:
            process = subprocess.Popen([python_cmd, "-m", "pip", "install", "--user", "-r", "requirements-glitch.txt"],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            output = stdout.decode() if stdout else stderr.decode()
            print("Installation output: {0}".format(output))
            
            if process.returncode == 0:
                print("Successfully installed dependencies from requirements-glitch.txt")
                return True
            else:
                print("Failed to install using requirements-glitch.txt")
        except Exception as e:
            print("Error during installation: {0}".format(str(e)))
    
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
            process = subprocess.Popen([python_cmd, "-m", "pip", "install", "--user", package],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            output = stdout.decode() if stdout else stderr.decode()
            
            if process.returncode == 0:
                print("Installed {0}".format(package))
            else:
                print("Failed to install {0}: {1}".format(package, output))
        except Exception as e:
            print("Error installing {0}: {1}".format(package, str(e)))
    
    return True

def main():
    """Main entry point for the script."""
    print("Starting Glitch setup for Jarvis...")
    
    version = check_python_version()
    if hasattr(version, 'major'):  # Python 2.7 has version as tuple
        if version.major < 3:
            print("Warning: Using Python 2.7. Will attempt to use Python 3 for installs.")
    else:
        if version[0] < 3:
            print("Warning: Using Python 2.7. Will attempt to use Python 3 for installs.")
    
    setup_glitch_environment()
    install_dependencies()
    
    print("\nSetup complete. If you're still experiencing issues:")
    print("1. Check Glitch logs for detailed error messages")
    print("2. Make sure your environment variables are set in the Glitch project settings")
    print("3. Try running 'python3 api.py' manually to see specific errors")
    print("4. Make sure Glitch is using Python 3 by checking the 'Tools' -> 'Terminal' output")

if __name__ == "__main__":
    main()
