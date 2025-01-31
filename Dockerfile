# Dockerfile

# Use Python 3.10 (to match cog.yaml's python_version)
FROM python:3.10-slim

# Set the working directory
WORKDIR /workspace

# Copy the application files into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0

# Upgrade pip
RUN pip install --upgrade pip

# Install torch and torchvision first
RUN pip install --no-cache-dir torch>=2.0 torchvision

# Install Cog (for both local usage of `cog predict` and any runtime needs)
# Install ALL Python dependencies explicitly
RUN pip install --no-cache-dir \
    transformers>=4.25.1 \
    diffusers>=0.14.0 \
    safetensors>=0.3.0 \
    accelerate>=0.14.0 \
    xformers \
    ftfy \
    tqdm \
    Pillow \
    opencv-python-headless \
    tensorboard \
    fastapi \
    uvicorn \
    requests \
    pyuploadcare \
    bitsandbytes \
    python-multipart \
    cog

# (Optional) If you have a separate requirements.txt for your dreambooth or main.py
# Make sure it includes fastapi, uvicorn, pyuploadcare, etc. if you need them.
# For example:
RUN pip install --no-cache-dir -r dreambooth/requirements.txt

# Expose port 8080 for your server in case you run a FastAPI app
EXPOSE 8080

# By default, run the FastAPI server.
# If you have main.py that starts uvicorn with app in main.py:app, do:
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# NOTE:
# - If you want to do `cog predict ...` instead of running the server,
#   override the default CMD at runtime:
#   docker run --gpus all -it --rm <your-image> cog predict ...
