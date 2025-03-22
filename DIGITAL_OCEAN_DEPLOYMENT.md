# Deploying Jarvis AI Assistant to Digital Ocean

This guide provides step-by-step instructions for deploying the Jarvis AI Assistant to Digital Ocean.

## Prerequisites

- A Digital Ocean account
- Docker installed on your local machine
- Digital Ocean CLI (doctl) installed and authenticated (optional, but recommended)
- Your API keys for Wit.ai and Google Gemini

## Option 1: Deploy to a Digital Ocean Droplet

### Step 1: Create a Droplet

1. Log in to your Digital Ocean account
2. Click "Create" and select "Droplets"
3. Choose an image: Ubuntu 22.04 LTS x64
4. Select a plan: Basic (Recommended: $12/mo with 2GB RAM, 1 CPU)
5. Choose a datacenter region close to your users
6. Add your SSH keys
7. Click "Create Droplet"

### Step 2: Connect to Your Droplet

```bash
ssh root@your_droplet_ip
```

### Step 3: Install Docker and Docker Compose

```bash
# Update package lists
apt update

# Install required packages
apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -

# Add Docker repository
add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Install Docker
apt update
apt install -y docker-ce

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version
```

### Step 4: Clone Your Repository or Upload Files

Option A: Clone from Git (if your repo is on GitHub/GitLab/etc.)
```bash
git clone https://your-repository-url.git
cd jarvis
```

Option B: Upload files using SCP
```bash
# Run this on your local machine
scp -r /path/to/jarvis root@your_droplet_ip:/root/jarvis
```

### Step 5: Create .env File with API Keys

```bash
cd jarvis
cat > .env << EOL
WIT_EN_KEY=your_wit_key_here
GOOGLE_API_KEY=your_google_key_here
EOL
```

### Step 6: Deploy with Docker Compose

```bash
docker-compose up -d
```

### Step 7: Configure Firewall (Optional but Recommended)

```bash
# Allow SSH, HTTP, and your app port
ufw allow 22
ufw allow 80
ufw allow 8000
ufw enable
```

### Step 8: Set Up Nginx as Reverse Proxy (Optional)

```bash
# Install Nginx
apt install -y nginx

# Create Nginx configuration
cat > /etc/nginx/sites-available/jarvis << EOL
server {
    listen 80;
    server_name your_domain.com;  # Replace with your domain or IP

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOL

# Enable the site
ln -s /etc/nginx/sites-available/jarvis /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

## Option 2: Deploy to Digital Ocean App Platform

### Step 1: Prepare Your Repository

Ensure your repository includes:
- Dockerfile
- requirements-prod.txt
- .dockerignore

### Step 2: Create a New App

1. Log in to your Digital Ocean account
2. Go to "Apps" in the left sidebar
3. Click "Create App"
4. Select your source (GitHub, GitLab, or upload directly)
5. Select your repository and branch
6. Click "Next"

### Step 3: Configure Your App

1. Verify that Digital Ocean detected your Dockerfile
2. Set environment variables:
   - WIT_EN_KEY: your_wit_key_here
   - GOOGLE_API_KEY: your_google_key_here
3. Choose your plan (Basic or Pro)
4. Click "Next"

### Step 4: Review and Launch

1. Review your app configuration
2. Click "Create Resources"
3. Wait for deployment to complete

### Step 5: Access Your App

Once deployment is complete, you can access your app at the provided URL.

## Monitoring and Maintenance

### View Logs

```bash
# For Droplet deployments
docker-compose logs -f

# For App Platform
# View logs in the Digital Ocean dashboard
```

### Update Your Deployment

```bash
# For Droplet deployments
git pull  # If using Git
docker-compose down
docker-compose up -d

# For App Platform
# Push to your repository, Digital Ocean will automatically redeploy
```

### Scale Your Application

For App Platform:
1. Go to your app in the Digital Ocean dashboard
2. Click "Settings" > "Resources"
3. Adjust the number of containers or container size

For Droplet:
Consider upgrading to a larger droplet or implementing a load balancer with multiple droplets.

## Troubleshooting

1. **App not starting**: Check logs for errors
2. **API keys not working**: Verify environment variables are set correctly
3. **Connection issues**: Check firewall settings and network configuration
4. **Performance issues**: Monitor resource usage and scale as needed
