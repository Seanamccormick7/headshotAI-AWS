# predownload_model.py
import os
import time
import logging
from huggingface_hub import snapshot_download, login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_id="stabilityai/stable-diffusion-3-medium-diffusers", max_retries=5):
    """
    Downloads and caches the model files with retry logic for better reliability.
    """
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable progress bars for cleaner logs
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"     # 10 minute timeout
    
    # Get token from environment variable
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN environment variable not set. Attempting anonymous download.")
    else:
        logger.info("Found HF_TOKEN in environment, logging in...")
        login(token=token)

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to download model {model_id}")
            cache_dir = snapshot_download(
                repo_id=model_id,
                resume_download=True,
                local_files_only=False,
                token=token,  # Pass token here as well for authentication
            )
            logger.info(f"Successfully downloaded model to {cache_dir}")
            return cache_dir
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 30 * (2 ** attempt)  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed. Could not download the model.")
                raise

if __name__ == "__main__":
    try:
        # Set a longer timeout for reliable downloads
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
        
        logger.info("Starting model download and caching process...")
        download_model()
        
        # We'll skip actually loading the model to save build time
        logger.info("Model successfully cached!")
    except Exception as e:
        logger.error(f"Download failed with error: {e}")
        # Don't raise the error to allow the build to continue
        # The model will be downloaded during runtime instead
        logger.info("Will attempt to download model at runtime instead.")