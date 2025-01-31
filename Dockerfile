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
    libglib2.0-0 \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install torch and torchvision with CUDA 11.8 support
RUN pip install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Force a compatible NumPy version (i.e., <2)
RUN pip install --no-cache-dir "numpy<2"

# Install the rest of the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r dreambooth/requirements.txt

# **Install the 'cog' Python module from GitHub**
RUN pip install --no-cache-dir git+https://github.com/replicate/cog.git

# Install Cog separately to ensure it's available for build and runtime
RUN sh -c "INSTALL_DIR=\"/usr/local/bin\" SUDO=\"\" $(curl -fsSL https://cog.run/install.sh)"

# Expose port 8080 for your server in case you run a FastAPI app
EXPOSE 8080

# By default, run the FastAPI server.
# If you have main.py that starts uvicorn with app in main.py:app, do:
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

# NOTE:
# - If you want to do `cog predict ...` instead of running the server,
#   override the default CMD at runtime:
#   docker run --gpus all -it --rm <your-image> cog predict ...
