FROM python:3.9-slim

# Set the working directory
WORKDIR /workspace

# Copy application files
COPY ./workspace /workspace

# Install dependencies
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Set the default command
CMD ["cog", "predict"]
