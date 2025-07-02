# Use the official Python image
FROM python:3.12

# Install system dependencies (adjust according to your program's needs)
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy code and related files
COPY . /app

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default startup command
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.enableCORS=false"]
