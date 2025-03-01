#!/bin/bash
# Script to install Python 3 on Glitch

echo "Attempting to install/upgrade Python 3 on Glitch..."

# Check current Python versions
echo "Current Python versions:"
python --version
python3 --version 2>/dev/null || echo "Python 3 not found"

# Update package lists
echo "Updating package lists..."
apt-get update

# Install Python 3 and pip
echo "Installing Python 3 and pip..."
apt-get install -y python3 python3-pip

# Verify installation
echo "Verifying Python 3 installation:"
python3 --version
pip3 --version

# Create symbolic links (if python3 exists but python doesn't point to it)
echo "Setting up Python 3 as default if needed..."
if [ -e /usr/bin/python3 ] && [ "$(readlink -f /usr/bin/python)" != "$(readlink -f /usr/bin/python3)" ]; then
  echo "Creating symbolic link for python -> python3"
  ln -sf /usr/bin/python3 /usr/bin/python
fi

if [ -e /usr/bin/pip3 ] && [ "$(readlink -f /usr/bin/pip)" != "$(readlink -f /usr/bin/pip3)" ]; then
  echo "Creating symbolic link for pip -> pip3"
  ln -sf /usr/bin/pip3 /usr/bin/pip
fi

# Create/update .python-version file
echo "3.8.10" > .python-version
echo "Created .python-version file"

# Update package.json if it exists
if [ -f package.json ]; then
  echo "Updating package.json to use python3..."
  sed -i 's/"start": "python /"start": "python3 /g' package.json
fi

echo "Python 3 setup complete. You may need to refresh or restart your Glitch project."
echo "Try running 'python3 --version' to confirm Python 3 is working."
