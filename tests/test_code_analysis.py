"""
Test script for the enhanced code analysis API.
Tests various scenarios and features of the code analysis endpoint.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from api import app
import json
from typing import Dict, Any
import numpy as np
from src.utils.voice_processor import VoiceProcessor

# We don't need async for TestClient - it's already handling async operations internally
@pytest.fixture
def client():
    return TestClient(app)

def test_python_analysis(client):
    """Test Python code analysis with all features."""
    code = """
def calculate_fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        for item in self.data:
            if isinstance(item, dict):
                item['processed'] = True
            elif isinstance(item, list):
                item.append('processed')
        return self.data
"""

    response = client.post(
        "/code/analyze",
        json={
            "code": code,
            "language": "python",
            "analysis_type": "full"
        }
    )
    
    assert response.status_code == 200, f"Failed with status {response.status_code}"
    result = response.json()
    
    # Verify basic response structure
    assert result["success"] == True
    assert result["language"] == "python"
    assert isinstance(result["suggestions"], list)
    assert isinstance(result["complexity_score"], (int, float))
    
    # Verify metrics
    assert "metrics" in result
    metrics = result["metrics"]
    assert metrics["total_lines"] > 0
    assert metrics["non_empty_lines"] > 0
    assert metrics["average_line_length"] > 0
    
    print("Python analysis test passed")

def test_invalid_code(client):
    """Test handling of invalid code."""
    code = """
def broken_function(
    print("This is broken")
"""

    response = client.post(
        "/code/analyze",
        json={
            "code": code,
            "language": "python",
            "analysis_type": "full"
        }
    )
    
    result = response.json()
    assert result["success"] == False
    assert "error" in result
    
    print("Invalid code test passed")

def test_security_analysis(client):
    """Test security analysis features."""
    code = """
import os
import sqlite3

def process_user_input(user_input):
    # Unsafe command execution
    os.system(user_input)
    
    # SQL Injection vulnerability
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM users WHERE name = '{user_input}'")
"""

    response = client.post(
        "/code/analyze",
        json={
            "code": code,
            "language": "python",
            "analysis_type": "security"
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["success"] == True
    assert result["security_issues"] is not None
    assert len(result["security_issues"]) > 0  # Should detect the security issues
    
    print("Security analysis test passed")

def test_best_practices(client):
    """Test best practices analysis."""
    code = """
# No type hints
def process_data(data):
    l = []  # Bad variable name
    for i in data:  # Non-descriptive loop variable
        if type(i) == str:  # Using type() instead of isinstance()
            l.append(i.upper())
    return l
"""

    response = client.post(
        "/code/analyze",
        json={
            "code": code,
            "language": "python",
            "analysis_type": "best_practices"
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    assert result["success"] == True
    assert result["best_practices"] is not None
    assert len(result["best_practices"]) > 0  # Should suggest improvements
    
    print("Best practices test passed")

def test_javascript_analysis(client):
    """Test JavaScript code analysis."""
    code = """
async function fetchUserData(userId) {
    try {
        const response = await fetch(`/api/users/${userId}`);
        if (!response.ok) {
            throw new Error('Failed to fetch user data');
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

class UserManager {
    constructor() {
        this.users = new Map();
    }

    addUser(user) {
        if (!user.id || !user.name) {
            throw new Error('Invalid user data');
        }
        this.users.set(user.id, user);
    }

    getUser(id) {
        return this.users.get(id);
    }
}
"""

    response = client.post(
        "/code/analyze",
        json={
            "code": code,
            "language": "javascript",
            "analysis_type": "full"
        }
    )
    
    assert response.status_code == 200, f"Failed with status {response.status_code}"
    result = response.json()
    
    # Verify analysis results
    assert result["success"] is True
    assert result["language"] == "javascript"
    assert isinstance(result["analysis"], dict)
    assert len(result["suggestions"]) > 0
    
    # Check for specific JavaScript analysis
    analysis = result["analysis"]
    assert "complexity" in analysis
    assert isinstance(analysis["complexity"], (int, float))
    assert analysis["complexity"] >= 1.0

    print("JavaScript analysis test passed")

def test_gemini_flash_capabilities(client):
    """Test enhanced multi-modal analysis capabilities"""
    test_code = '''def sum(a, b): return a + b'''
    response = client.post(
        "/code/analyze",
        json={
            "code": test_code,
            "language": "python",
            "analysis_type": "full"
        }
    )
    
    # Validate enhanced metrics
    result = response.json()
    assert result["success"] == True
    assert "security_analysis" in result["analysis"]
    assert result["complexity_score"] > 0
    assert "type_hints" in result["suggestions"]

def test_audio_processing_compatibility():
    """Test audio processing with Gemini voice capabilities"""
    try:
        # Initialize processor with default settings
        processor = VoiceProcessor()
        
        # Create test audio data (1 second of silence)
        duration = 1  # seconds
        sample_rate = 16000
        num_samples = duration * sample_rate
        audio_data = np.zeros(num_samples, dtype=np.int16)
        
        # Process audio
        processed = processor.process_audio(audio_data.tobytes())
        
        # Verify output format and content
        assert processed is not None, "Processed audio should not be None"
        assert len(processed) > 0, "Processed audio should not be empty"
        assert processed.startswith(b'RIFF'), "Invalid WAV header"
        assert b'fmt ' in processed, "Missing format chunk"
        
    except Exception as e:
        pytest.fail(f"Audio processing test failed: {str(e)}")
