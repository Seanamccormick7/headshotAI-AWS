# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /workspace

# Copy the application files
COPY ./ ./

# Navigate to the correct directory and install dependencies
RUN pip install --no-cache-dir -r dreambooth/requirements.txt

# Expose port 8080
EXPOSE 8080

# Set the default command
CMD ["cog", "predict"]
