# Rice Price Prediction - Docker Deployment

## Docker Image Details

- **Base Image**: python:3.13-slim (multi-stage build)
- **Final Image Size**: ~1.3GB
- **Architecture**: Optimized with multi-stage build
- **Security**: Runs as non-root user (appuser:1000)

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Using Docker CLI

```bash
# Build the image
docker build -t rice-price-prediction:latest .

# Run the container
docker run -d \
  -p 8001:8000 \
  -v $(pwd)/models:/app/models:ro \
  --name rice-price-api \
  rice-price-prediction:latest

# View logs
docker logs -f rice-price-api

# Stop and remove
docker stop rice-price-api && docker rm rice-price-api
```

## API Endpoints

Once running, access the API at `http://localhost:8001`:

- **Interactive Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health
- **Root**: http://localhost:8001/
- **Predict**: POST http://localhost:8001/predict

## Example API Request

```bash
curl -X POST "http://localhost:8001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "country_code": "BGD",
    "county": "Dhaka",
    "subcounty": "Dhaka",
    "latitude": 23.81,
    "longitude": 90.41,
    "commodity": "Rice (coarse, BR-8/ 11/, Guti Sharna)",
    "price_flag": "actual",
    "price_type": "Wholesale",
    "year": 2024,
    "month": 11,
    "day": 7
  }'
```

## Environment Variables

Configure via `docker-compose.yml` or `-e` flag:

- `PYTHONUNBUFFERED=1`: Enable real-time logging

## Volume Mounts

The container requires the trained models directory:

```yaml
volumes:
  - ./models:/app/models:ro  # Read-only models directory
```

## Resource Limits

Configured in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 2G
    reservations:
      cpus: '1'
      memory: 1G
```

## Docker Best Practices Implemented

✅ **Multi-stage build** - Separates build and runtime environments
✅ **Minimal base image** - Uses `python:3.13-slim`
✅ **Layer caching** - Optimized COPY order for faster rebuilds
✅ **Non-root user** - Runs as `appuser:1000` for security
✅ **Health checks** - Built-in container health monitoring
✅ **.dockerignore** - Excludes unnecessary files
✅ **No secrets** - Models/data mounted as volumes, not baked in
✅ **Resource limits** - Prevents resource exhaustion
✅ **Read-only volumes** - Models mounted as read-only

## Troubleshooting

### Port already in use
```bash
# Change port in docker-compose.yml
ports:
  - "8002:8000"  # Use different host port
```

### Model not found
```bash
# Ensure models directory exists and contains:
# - stacking_ensemble_model.pkl
# - label_encoders.pkl
ls -la models/
```

### Check container logs
```bash
docker-compose logs -f
```

### Access container shell
```bash
docker-compose exec api /bin/bash
```

## Production Deployment

For production, consider:

1. **Use a registry**: Push to Docker Hub, ECR, or GCR
2. **Kubernetes**: Deploy with Kubernetes for scaling
3. **Load balancer**: Add nginx or traefik
4. **Monitoring**: Add Prometheus + Grafana
5. **Secrets management**: Use Docker secrets or vault

## Development Workflow

```bash
# 1. Build image
docker-compose build

# 2. Start container
docker-compose up -d

# 3. Make code changes

# 4. Rebuild and restart
docker-compose up -d --build

# 5. View logs
docker-compose logs -f

# 6. Stop
docker-compose down
```
