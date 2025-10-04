# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all files to container
COPY . .

# Expose a port (e.g., if Streamlit or FastAPI app)
EXPOSE 8501

# Command to run your app (adjust if using Flask, FastAPI, etc.)
CMD ["python", "app.py"]
