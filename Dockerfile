# Dockerfile

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Make sure you have a system environment (Python 3.10 or 3.9, etc.)
RUN apt-get update && apt-get install -y git wget python3 python3-pip

WORKDIR /app
COPY . /app

# Install your DreamBooth + project dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r dreambooth/requirements.txt

# (Optionally) configure accelerate with default answers:
RUN accelerate config default

# Expose port, set the entrypoint, etc. (custom to your needs)
EXPOSE 8080
CMD ["python3", "main.py"]