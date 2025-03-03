# predownload_model.py
import os
import time
import logging
from huggingface_hub import snapshot_download
from diffusers import StableDiffusion3Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_id="stabilityai/stable-diffusion-3-medium-diffusers", max_retries=5):
    """
    Downloads and caches the model files with retry logic for better reliability.
    """
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable progress bars for cleaner logs
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"     # 10 minute timeout

    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} to download model {model_id}")
            cache_dir = snapshot_download(
                repo_id=model_id,
                resume_download=True,
                local_files_only=False,
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
        
        # Optionally load the model to ensure all components are cached
        logger.info("Initializing the model to verify caching...")
        SD3_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
        StableDiffusion3Pipeline.from_pretrained(SD3_ID, device_map="cpu")
        logger.info("Model successfully cached and verified!")
    except Exception as e:
        logger.error(f"Download failed with error: {e}")
        raise