# Dockerfile

# Use an official NVIDIA CUDA 11.8 runtime base (Ubuntu 22.04)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Ensure Python 3 + pip are installed
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git libgl1-mesa-glx libglib2.0-0 unzip \
 && rm -rf /var/lib/apt/lists/*

# If you want python -> python3 symlink
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
COPY . .

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch and Triton
RUN python -m pip install --no-cache-dir torch==2.1.0+cu118 torchvision==0.16.0+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118 && \
    python -m pip install --no-cache-dir triton==2.1.0

# Force NumPy <2
RUN python -m pip install --no-cache-dir "numpy<2"

# Install the rest of your Python deps
RUN python -m pip install --no-cache-dir -r dreambooth/requirements.txt

#forcing triton to be installed with 2.1.0
RUN python -m pip install --no-cache-dir triton==2.1.0

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
