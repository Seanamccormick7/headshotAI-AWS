# Dockerfile

# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /workspace

# Copy all application files into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install torch and torchvision with CUDA 11.8 support
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install a compatible version of NumPy
RUN pip install --no-cache-dir "numpy<2"

# Install other Python dependencies (including Celery and Redis)
RUN pip install --no-cache-dir -r dreambooth/requirements.txt

# Expose port 8080 for uvicorn
EXPOSE 8080

# By default, run the FastAPI web server. (For Celery, run a separate command.)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
