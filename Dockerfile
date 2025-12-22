# Use Python 3.9 as base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY recording.py .
COPY try.py .
COPY main_pipeline.py .
COPY collab\ files/ ./collab\ files/

# Create output directory
RUN mkdir -p /app/output

# Make scripts executable
RUN chmod +x main_pipeline.py

# Set default environment variables
ENV VIDEO_URL=http://172.17.13.231:5000/video_feed
ENV MAX_DURATION=3000
ENV OUTPUT_DIR=/app/output
ENV INPUT_VIDEO=flask_record.mp4
ENV OUTPUT_CSV=eye_metrics_output.csv
ENV INPUT_CSV=eye_metrics_output.csv
ENV DATA_FOLDER=new_folder
ENV INPUT_FILE=cleaned_file.csv
ENV OUTPUT_FILE=prediction_result.txt

# Expose port (if needed for any web interface)
EXPOSE 8080

# Default command
CMD ["python", "main_pipeline.py"] 