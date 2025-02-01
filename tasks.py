# tasks.py
import os
import tempfile
import zipfile
import requests
import subprocess
import glob
import shutil
import pyuploadcare

from celery_app import celery_app
from predict import run_training

# Configure Uploadcare keys
pyuploadcare.conf.api_secret = os.environ.get("UPLOADCARE_SECRET_KEY", "secret")
pyuploadcare.conf.api_public = os.environ.get("UPLOADCARE_PUBLIC_KEY", "public")

@celery_app.task(bind=True)
def generate_images_task(self, req_data: dict):
    """
    Celery task to process image generation.
    Downloads user images, runs training, uploads the generated images,
    calls back to Next.js, and then cleans up temporary data.
    """
    # For cleanup tracking:
    instance_dir = None
    output_dir = None
    try:
        userId = req_data.get("userId")
        instanceImages = req_data.get("instanceImages", [])
        callbackUrl = req_data.get("callbackUrl")
        
        # 1) Download user's instance images into a temporary directory.
        instance_dir = tempfile.mkdtemp(prefix="instance_images_")
        for i, uuid in enumerate(instanceImages):
            download_url = f"https://ucarecdn.com/{uuid}/-/format/jpeg/"
            out_path = os.path.join(instance_dir, f"{i}.jpg")
            r = requests.get(download_url)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"Failed to download {uuid}, status {r.status_code}")

        # 2) Zip the images
        zip_path = os.path.join(instance_dir, "images.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for img_file in os.listdir(instance_dir):
                if img_file.endswith(".jpg"):
                    zf.write(os.path.join(instance_dir, img_file), arcname=img_file)

        # 3) Create a temporary output directory for the training results.
        output_dir = tempfile.mkdtemp(prefix="trained_model_")
        # Use a small number of steps for testing (adjust as needed)
        instance_prompt = "photo of sks person"
        training_steps = 10

        # Run DreamBooth training. Note: run_training is updated to accept output_dir.
        train_output = run_training(
            instance_data=zip_path,
            instance_prompt=instance_prompt,
            steps=training_steps,
            output_dir=output_dir,
            sample_prompt="high quality professional headshot of sks person"
        )
        print("DreamBooth training output:", train_output)

        # 4) Upload generated images to Uploadcare.
        generated_uuids = []
        # Assume training writes images into a subfolder like "<output_dir>/10/samples"
        samples_dir = os.path.join(output_dir, "10", "samples")
        if not os.path.isdir(samples_dir):
            print("No images found in", samples_dir)
        else:
            from pyuploadcare import Uploadcare
            uploadcare = Uploadcare()
            for filename in os.listdir(samples_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    fullpath = os.path.join(samples_dir, filename)
                    with open(fullpath, "rb") as file_obj:
                        file_uploaded = uploadcare.upload(file_obj)
                    generated_uuids.append(str(file_uploaded.uuid))

        # 5) Callback to Next.js with the generated image UUIDs.
        callback_payload = {
            "userId": userId,
            "generatedUuids": generated_uuids,
        }
        resp = requests.post(callbackUrl, json=callback_payload)
        if not resp.ok:
            print(f"Callback to Next.js failed: {resp.text}")

        return {"message": "Generation task completed", "generatedUuids": generated_uuids}
    except Exception as e:
        print("Error in generate_images_task:", str(e))
        raise e
    finally:
        # Clean up temporary directories to free up space.
        if instance_dir and os.path.exists(instance_dir):
            shutil.rmtree(instance_dir, ignore_errors=True)
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)