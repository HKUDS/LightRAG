# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file from parent directory
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the palmier directory
COPY . ./palmier/

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8020

# Expose the port the app runs on
EXPOSE 8020

# Run the application
CMD ["python", "palmier/main.py"]
