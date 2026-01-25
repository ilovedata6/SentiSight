#!/bin/bash
# Deployment script for SentiSight

set -e

echo "========================================"
echo "SentiSight Deployment Script"
echo "========================================"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration"
    exit 1
fi

# Build and start services
echo "Building Docker images..."
docker-compose build

echo "Starting services..."
docker-compose up -d

echo "Waiting for services to be healthy..."
sleep 10

# Check health
echo "Checking API health..."
curl -f http://localhost:8000/health || { echo "⚠️  API not responding"; exit 1; }

echo ""
echo "========================================"
echo "✅ Deployment Complete!"
echo "========================================"
echo ""
echo "Services:"
echo "  - API:        http://localhost:8000"
echo "  - API Docs:   http://localhost:8000/docs"
echo "  - Frontend:   http://localhost:8501"
echo "  - Database:   localhost:5432"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo "========================================"
