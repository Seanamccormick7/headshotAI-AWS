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

# Install CUDA 12.1-compatible PyTorch
RUN python -m pip install --no-cache-dir torch==2.2.0+cu121 torchvision==0.16.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install bitsandbytes for CUDA 12.x
RUN python -m pip install https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/0.42.0/bitsandbytes-0.42.0-py3-none-any.whl

# Install xformers
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
RUN python -m pip install --no-cache-dir "numpy<2"
RUN python -m pip install --no-cache-dir -r dreambooth/requirements.txt

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]