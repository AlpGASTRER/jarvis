# Jarvis AI Assistant Deployment Guide

This guide covers deploying the Jarvis AI Assistant API in different environments.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- A server with at least:
  - 2 CPU cores
  - 4GB RAM
  - 20GB storage
- API Keys:
  - Wit.ai API key
  - Google API key (Gemini)

## Local Development Deployment

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

4. Create .env file:
```bash
echo WIT_EN_KEY=your_wit_key > .env
echo GOOGLE_API_KEY=your_google_key >> .env
```

5. Run the development server:
```bash
python api.py
```

## Production Deployment

### Docker Deployment

1. Create a Dockerfile:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run with Gunicorn
CMD ["gunicorn", "api:app", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "--workers", "4"]
```

2. Create docker-compose.yml:
```yaml
version: '3.8'

services:
  jarvis-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - WIT_EN_KEY=${WIT_EN_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
```

3. Build and run:
```bash
docker-compose up -d
```

### Cloud Deployment (AWS)

1. **EC2 Setup:**
   - Launch Ubuntu EC2 instance (t2.medium or better)
   - Configure security group:
     - HTTP (80)
     - HTTPS (443)
     - Custom TCP (8000)

2. **Install Dependencies:**
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv portaudio19-dev python3-pyaudio nginx
```

3. **Setup Application:**
```bash
# Clone repository
git clone [your-repo-url]
cd jarvis

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install gunicorn

# Setup environment variables
sudo nano /etc/environment
# Add:
# WIT_EN_KEY=your_wit_key
# GOOGLE_API_KEY=your_google_key
```

4. **Create Systemd Service:**
```bash
sudo nano /etc/systemd/system/jarvis.service
```
```ini
[Unit]
Description=Jarvis AI Assistant API
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/jarvis
Environment="PATH=/home/ubuntu/jarvis/venv/bin"
ExecStart=/home/ubuntu/jarvis/venv/bin/gunicorn api:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 --workers 4

[Install]
WantedBy=multi-user.target
```

5. **Configure Nginx:**
```bash
sudo nano /etc/nginx/sites-available/jarvis
```
```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

6. **Enable and Start Services:**
```bash
# Enable Nginx config
sudo ln -s /etc/nginx/sites-available/jarvis /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx

# Start Jarvis service
sudo systemctl start jarvis
sudo systemctl enable jarvis
```

7. **SSL Setup (Optional but Recommended):**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your_domain.com
```

### Kubernetes Deployment

1. Create kubernetes/deployment.yaml:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jarvis-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jarvis-api
  template:
    metadata:
      labels:
        app: jarvis-api
    spec:
      containers:
      - name: jarvis-api
        image: your-registry/jarvis-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: WIT_EN_KEY
          valueFrom:
            secretKeyRef:
              name: jarvis-secrets
              key: wit-key
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: jarvis-secrets
              key: google-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

2. Create kubernetes/service.yaml:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: jarvis-api
spec:
  selector:
    app: jarvis-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

3. Create kubernetes/secrets.yaml:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: jarvis-secrets
type: Opaque
data:
  wit-key: <base64-encoded-wit-key>
  google-key: <base64-encoded-google-key>
```

4. Deploy:
```bash
kubectl apply -f kubernetes/
```

## Monitoring and Maintenance

1. **Logging:**
   - Application logs: `/app/logs/`
   - Nginx logs: `/var/log/nginx/`
   - System logs: `journalctl -u jarvis`

2. **Monitoring:**
   - Set up Prometheus for metrics
   - Use Grafana for visualization
   - Monitor key metrics:
     - API response times
     - Memory usage
     - CPU usage
     - Error rates

3. **Backup:**
   - Regularly backup:
     - Configuration files
     - Environment variables
     - Custom models/data
     - Logs

4. **Updates:**
```bash
# Pull latest changes
git pull

# Update dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart jarvis
```

## Security Considerations

1. **API Security:**
   - Implement rate limiting
   - Add API key authentication
   - Enable CORS only for trusted domains
   - Use HTTPS in production

2. **Server Security:**
   - Keep system updated
   - Use UFW firewall
   - Implement fail2ban
   - Regular security audits

3. **Environment Variables:**
   - Never commit .env files
   - Use secrets management in production
   - Rotate API keys regularly

## Troubleshooting

1. **API Not Responding:**
   - Check service status: `sudo systemctl status jarvis`
   - Check logs: `journalctl -u jarvis -n 100`
   - Verify port availability: `netstat -tulpn | grep 8000`

2. **Voice Recognition Issues:**
   - Check Wit.ai API key
   - Verify audio device permissions
   - Check audio format compatibility

3. **High Resource Usage:**
   - Monitor with `top` or `htop`
   - Check worker configuration
   - Adjust resource limits

4. **WebSocket Connection Issues:**
   - Verify Nginx WebSocket configuration
   - Check client connection settings
   - Monitor connection logs

## Performance Optimization

1. **API Optimization:**
   - Adjust worker count based on CPU cores
   - Implement caching where appropriate
   - Optimize database queries

2. **Resource Management:**
   - Monitor and adjust resource limits
   - Use appropriate instance types
   - Implement auto-scaling

3. **Network Optimization:**
   - Use CDN for static assets
   - Optimize payload sizes
   - Enable compression

## Support

For issues and support:
1. Check the troubleshooting guide
2. Review logs for specific errors
3. Contact the development team
4. Submit issues on GitHub
