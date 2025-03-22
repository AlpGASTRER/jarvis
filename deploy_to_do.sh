#!/bin/bash
# Script to deploy Jarvis AI Assistant to Digital Ocean

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Jarvis AI Assistant - Digital Ocean Deployment Script${NC}"
echo "======================================================"

# Check if doctl is installed
if ! command -v doctl &> /dev/null; then
    echo -e "${RED}Error: doctl is not installed.${NC}"
    echo "Please install the Digital Ocean CLI (doctl) first:"
    echo "https://docs.digitalocean.com/reference/doctl/how-to/install/"
    exit 1
fi

# Check if user is authenticated with doctl
if ! doctl account get &> /dev/null; then
    echo -e "${RED}Error: You are not authenticated with Digital Ocean.${NC}"
    echo "Please run 'doctl auth init' and follow the instructions."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found.${NC}"
    echo "Creating a template .env file. Please edit it with your API keys."
    echo "WIT_EN_KEY=your_wit_key_here" > .env
    echo "GOOGLE_API_KEY=your_google_key_here" >> .env
    echo -e "${YELLOW}A template .env file has been created. Please edit it with your API keys and run this script again.${NC}"
    exit 1
fi

# Deployment options
echo "Please select a deployment option:"
echo "1) Deploy to Digital Ocean App Platform"
echo "2) Deploy to a Digital Ocean Droplet"
read -p "Enter your choice (1 or 2): " choice

case $choice in
    1)
        echo -e "${GREEN}Deploying to Digital Ocean App Platform...${NC}"
        
        # Check if .do/app.yaml exists
        if [ ! -f .do/app.yaml ]; then
            echo -e "${RED}Error: .do/app.yaml file not found.${NC}"
            echo "Please make sure you have the .do/app.yaml file in your project."
            exit 1
        fi
        
        # Deploy using doctl
        echo "Deploying app..."
        doctl apps create --spec .do/app.yaml
        
        echo -e "${GREEN}Deployment initiated!${NC}"
        echo "You can check the status in the Digital Ocean dashboard."
        ;;
        
    2)
        echo -e "${GREEN}Preparing for Droplet deployment...${NC}"
        
        # Create a deployment package
        echo "Creating deployment package..."
        tar -czf jarvis-deployment.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='venv' .
        
        echo -e "${GREEN}Deployment package created: jarvis-deployment.tar.gz${NC}"
        echo ""
        echo "To deploy to a Droplet, follow these steps:"
        echo "1. Create a Droplet in the Digital Ocean dashboard"
        echo "2. Upload the deployment package to your Droplet:"
        echo "   scp jarvis-deployment.tar.gz root@your-droplet-ip:~/"
        echo "3. SSH into your Droplet:"
        echo "   ssh root@your-droplet-ip"
        echo "4. Extract and deploy:"
        echo "   mkdir -p jarvis && tar -xzf jarvis-deployment.tar.gz -C jarvis"
        echo "   cd jarvis && docker-compose up -d"
        ;;
        
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}Done!${NC}"
