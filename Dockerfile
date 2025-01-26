# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /workspace

# Copy the application files into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r dreambooth/requirements.txt

# Expose port (optional, depending on your use case)
EXPOSE 8080

# Set the default command to use cog
CMD ["cog", "predict"]
