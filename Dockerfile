# Dockerfile

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git libgl1-mesa-glx libglib2.0-0 unzip \
    build-essential ninja-build \
 && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /workspace
COPY . .

RUN python -m pip install --upgrade pip

# Install PyTorch and torchvision with matching CUDA version
RUN pip3 install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install bitsandbytes for CUDA 12.x
RUN python -m pip install bitsandbytes==0.41.1

# Install xformers with matching CUDA support
RUN pip install -U xformers==0.0.22.post7

# Install hf_transfer for faster downloads
RUN pip install hf_transfer

# Install remaining requirements
RUN python -m pip install --no-cache-dir "numpy<2"
RUN python -m pip install --no-cache-dir -r dreambooth/requirements.txt

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]