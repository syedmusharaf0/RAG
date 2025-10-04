# Use official Python base image
FROM python:3.10-slim

# Prevent Python from buffering output (helpful for logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (useful for FAISS, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to container
COPY . .

# Expose the port your app uses (e.g., Streamlit or FastAPI)
EXPOSE 8501

# Command to run your app
CMD ["python", "app.py"]
