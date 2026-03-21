FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies first (to cache the layer)
# Using our lightweight docker requirements
COPY requirements-docker.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --default-timeout=2000 --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/


# Create output directories to avoid permission issues when volume mounting
RUN mkdir -p /app/output/uploads

# We do not set ENTRYPOINT here so that docker-compose can override the command for the API and Consumer independently
EXPOSE 8000
