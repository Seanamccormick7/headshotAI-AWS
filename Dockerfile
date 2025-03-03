# Dockerfile

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y git wget python3 python3-pip python3-venv unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Create directories for persistent storage
RUN mkdir -p /app/models /app/class_images /app/temp

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Diffusers from source
RUN pip3 install --upgrade pip
RUN pip3 install git+https://github.com/huggingface/diffusers.git

# Install hf_transfer for faster model downloads
RUN pip3 install hf_transfer

# Install the project dependencies
RUN pip3 install -r dreambooth/requirements.txt

# Create accelerate config file directly
RUN mkdir -p /root/.cache/huggingface/accelerate
RUN echo '{\n\
  "compute_environment": "LOCAL_MACHINE",\n\
  "distributed_type": "NO",\n\
  "downcast_bf16": "no",\n\
  "gpu_ids": "all",\n\
  "machine_rank": 0,\n\
  "main_training_function": "main",\n\
  "mixed_precision": "fp16",\n\
  "num_machines": 1,\n\
  "num_processes": 1,\n\
  "rdzv_backend": "static",\n\
  "tpu_env": [],\n\
  "tpu_use_cluster": false,\n\
  "tpu_use_sudo": false,\n\
  "use_cpu": false\n\
}' > /root/.cache/huggingface/accelerate/default_config.yaml

# Set environment variables to use our persistent directories
ENV TMPDIR="/app/temp"

# Expose port, set the entrypoint, etc. (custom to your needs)
EXPOSE 8080
CMD ["python3", "main.py"]