# tasks.py
import os
import tempfile
import zipfile
import requests
import subprocess
import glob
import shutil
import pyuploadcare
import torch
import logging

from celery_app import celery_app
from predict import run_training
from inference import run_inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Uploadcare keys (ensure these environment variables are set)
pyuploadcare.conf.api_secret = os.environ.get("UPLOADCARE_SECRET_KEY", "secret")
pyuploadcare.conf.api_public = os.environ.get("UPLOADCARE_PUBLIC_KEY", "public")

# Define permanent directories for model outputs
MODELS_DIR = '/app/models'  # This should be a mounted volume
os.makedirs(MODELS_DIR, exist_ok=True)

def get_disk_free_space_gb(path):
    """Get free space in GB for the file system containing path."""
    stat = os.statvfs(path)
    free_space = stat.f_frsize * stat.f_bavail
    return free_space / (1024 ** 3)  # Convert to GB

def clean_up_all_files(user_model_dir):
    """Completely delete all files related to this job"""
    try:
        logger.info(f"Cleaning up all files in {user_model_dir}")
        if os.path.exists(user_model_dir):
            shutil.rmtree(user_model_dir)
            logger.info(f"Successfully deleted {user_model_dir}")
            
        # Also clean out temporary directories
        for temp_dir in ['/tmp', '/app/temp']:
            if os.path.exists(temp_dir):
                for item in os.listdir(temp_dir):
                    item_path = os.path.join(temp_dir, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        logger.warning(f"Error deleting {item_path}: {e}")
                        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        logger.info("Cleanup completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False

@celery_app.task(bind=True)
def generate_images_task(self, req_data: dict):
    """
    Celery task to process image generation.
    Downloads user images, runs training, uploads the generated images,
    calls back to Next.js, and then cleans up ALL data.
    """
    instance_dir = None
    output_dir = None
    user_model_dir = None
    
    # Check disk space before starting
    free_space = get_disk_free_space_gb(MODELS_DIR)
    logger.info(f"Current free disk space: {free_space:.2f}GB")
    
    try:
        userId = req_data.get("userId")
        instanceImages = req_data.get("instanceImages", [])
        callbackUrl = req_data.get("callbackUrl")

        # Create a user-specific directory for output
        user_model_dir = os.path.join(MODELS_DIR, f"user_{userId}")
        os.makedirs(user_model_dir, exist_ok=True)

        # 1) Download instance images
        instance_dir = os.path.join(user_model_dir, "instance_images")
        os.makedirs(instance_dir, exist_ok=True)
        
        for i, uuid in enumerate(instanceImages):
            download_url = f"https://ucarecdn.com/{uuid}/-/format/jpeg/"
            out_path = os.path.join(instance_dir, f"{i}.jpg")
            r = requests.get(download_url)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
            else:
                logger.error(f"Failed to download {uuid}, status {r.status_code}")

        # 2) Zip them
        zip_path = os.path.join(instance_dir, "images.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for img_file in os.listdir(instance_dir):
                if img_file.endswith(".jpg"):
                    zf.write(
                        os.path.join(instance_dir, img_file),
                        arcname=img_file
                    )

        # 3) Build dynamic prompt info
        gender = req_data.get("gender", "person")
        age = req_data.get("age", None)
        age_str = f"{age}-year-old " if age is not None else ""
        hairColor = req_data.get("hairColor", "unknown hair color")
        hairLength = req_data.get("hairLength", "unknown length")
        ethnicity = req_data.get("ethnicity", "")
        bodyType = req_data.get("bodyType", "")
        attire = req_data.get("attire", "")
        backgrounds = req_data.get("backgrounds", "")
        glasses = req_data.get("glasses", False)
        glasses_str = "wearing glasses" if glasses else "no glasses"
        
        # Use a unique token instead of "sks" for better subject identification
        unique_token = "prs123"  # Choose any unique token
        
        # class_prompt - keep simple
        class_prompt = "Photo of a person"

        # instance_prompt - use unique token
        instance_prompt = (
            f"Photo of a {unique_token} {age_str}{gender} with {hairColor} hair "
            f"({hairLength}), of {ethnicity} ethnicity, {bodyType} build, {glasses_str}."
        )

        # Inference_prompt with specific pose guidance
        inference_prompt = (
            f"Professional studio headshot of the {unique_token} {age_str}{gender} with {hairColor} hair "
            f"({hairLength}), {ethnicity} ethnicity, {bodyType} build, {glasses_str}, "
            f"wearing {attire} clothing, in {backgrounds}, shot with professional lighting, "
            f"high detail, 4k, sharp focus, DSLR, professional portrait photography, "
            f"high-end editorial photography, trending on artstation, highly detailed"
        )

        # 4) Set training steps
        training_steps = 1200  

        # 5) Create output directory - use the mounted volume
        output_dir = os.path.join(user_model_dir, "trained_model")
        os.makedirs(output_dir, exist_ok=True)
        
        # Use the existing class_images directory
        class_data_dir = "class_images"  # This references your pre-existing folder

        # 6) DreamBooth training
        train_output = run_training(
            instance_data=zip_path,
            instance_prompt=instance_prompt,
            inference_prompt=inference_prompt,
            steps=training_steps,
            output_dir=output_dir,
            class_prompt=class_prompt,
            class_data_dir=class_data_dir,
            num_class_images=50,
        )
        logger.info(f"DreamBooth training output: {train_output}")
        logger.info(f"Instance prompt: {instance_prompt}")
        logger.info(f"Inference prompt: {inference_prompt}")

        # 7) Post-training inference
        inference_dir = os.path.join(output_dir, "inference_output")
        run_inference(
            base_model=output_dir,
            prompt=inference_prompt,
            outdir=inference_dir,
            num_images=5,  # doing 5 just for testing, change to 100 in production
            guidance_scale=7.5,
            num_inference_steps=40,
            torch_dtype=torch.float16,
        )

        # 8) Upload generated images to Uploadcare
        generated_uuids = []
        if not os.path.isdir(inference_dir):
            logger.error(f"No inference images found in {inference_dir}")
        else:
            from pyuploadcare import Uploadcare
            uploadcare = Uploadcare()
            for filename in os.listdir(inference_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    fullpath = os.path.join(inference_dir, filename)
                    with open(fullpath, "rb") as file_obj:
                        file_uploaded = uploadcare.upload(file_obj)
                    generated_uuids.append(str(file_uploaded.uuid))

        # 9) Callback with the final images
        callback_payload = {
            "userId": userId,
            "generatedUuids": generated_uuids,
        }
        resp = requests.post(callbackUrl, json=callback_payload)
        if not resp.ok:
            logger.error(f"Callback to Next.js failed: {resp.text}")

        # SUCCESS - All done, return result
        return {"message": "Generation task completed", "generatedUuids": generated_uuids}

    except Exception as e:
        logger.error(f"Error in generate_images_task: {str(e)}")
        raise e
    finally:
        # CRITICAL: Clean up EVERYTHING - we don't need to keep any files
        if user_model_dir:
            clean_up_all_files(user_model_dir)
            
        # Log the available disk space after cleanup
        free_space = get_disk_free_space_gb(MODELS_DIR)
        logger.info(f"Disk space after cleanup: {free_space:.2f}GB")