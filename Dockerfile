FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libasound2-dev \
    portaudio19-dev \
    libpulse-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080

# Run the application
CMD exec uvicorn api:app --host 0.0.0.0 --port ${PORT}
