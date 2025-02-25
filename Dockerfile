# Dockerfile

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y git wget python3 python3-pip python3-venv unzip

WORKDIR /app
COPY . /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install your DreamBooth + project dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r dreambooth/requirements.txt

# (Optionally) configure accelerate with default answers:
RUN accelerate config default --mixed_precision=fp16

# Expose port, set the entrypoint, etc. (custom to your needs)
EXPOSE 8080
CMD ["python3", "main.py"]