---
applyTo: "Dockerfile"
---

# Dockerfile Guidelines

## Base Image

- Use official Python slim images for smaller image sizes
- Pin Python version to ensure consistency
- Current: `python:3.11-slim`

## Layer Optimization

- Group related RUN commands to reduce layers
- Clean up package manager caches in the same layer as installation
- Order layers from least to most frequently changing (base image → system packages → dependencies → application code)

## Dependencies

- Install system dependencies (like ffmpeg) before Python packages
- Use `--no-install-recommends` to minimize installed packages
- Remove package manager caches with `rm -rf /var/lib/apt/lists/*`
- Use `--no-cache-dir` with pip to avoid caching packages in the image

## Application Setup

- Set WORKDIR early in the Dockerfile
- Copy requirements.txt before application code to leverage layer caching
- Copy application code last to maximize cache hits during development

## Runtime Configuration

- Set PYTHONUNBUFFERED=1 to see logs in real-time
- Expose necessary ports (currently using 9001)
- Use CMD with exec form (JSON array) for proper signal handling
- Configure appropriate number of workers based on application needs

## Security

- Run as non-root user when possible (consider adding in future updates)
- Don't include sensitive data in the image
- Keep base images updated for security patches
