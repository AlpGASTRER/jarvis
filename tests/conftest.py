"""
Pytest configuration file for test discovery and path setup.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure pytest for async tests
pytest_plugins = ["pytest_asyncio"]
