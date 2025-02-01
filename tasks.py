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

# Configure Uploadcare keys (ensure these environment variables are set)
pyuploadcare.conf.api_secret = os.environ.get("UPLOADCARE_SECRET_KEY", "secret")
pyuploadcare.conf.api_public = os.environ.get("UPLOADCARE_PUBLIC_KEY", "public")

@celery_app.task(bind=True)
def generate_images_task(self, req_data: dict):
    """
    Celery task to process image generation.
    """
    try:
        # Extract request data
        userId = req_data.get("userId")
        # (Other user attributes are available if you wish to use them.)
        instanceImages = req_data.get("instanceImages", [])
        callbackUrl = req_data.get("callbackUrl")

        # 1) Download user’s instance images from Uploadcare
        instance_dir = tempfile.mkdtemp()
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

        # 3) Run DreamBooth training (this is the long-running part)
        # For demonstration, we use a dummy instance prompt and a small number of steps.
        instance_prompt = "photo of sks person"  # You can build a more detailed prompt here.
        training_steps = 10  # Adjust for testing or production

        train_output = run_training(
            instance_data=zip_path,
            instance_prompt=instance_prompt,
            steps=training_steps,
            sample_prompt="high quality professional headshot of sks person"
        )
        print("DreamBooth training output:", train_output)

        # 4) Upload generated images to Uploadcare
        generated_uuids = []
        # Ensure this path matches where your training script writes sample images.
        output_images_dir = "trained_model/10/samples"  

        if not os.path.isdir(output_images_dir):
            print("No images found in", output_images_dir)
        else:
            from pyuploadcare import Uploadcare
            uploadcare = Uploadcare()
            for filename in os.listdir(output_images_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    fullpath = os.path.join(output_images_dir, filename)
                    with open(fullpath, "rb") as file_obj:
                        file_uploaded = uploadcare.upload(file_obj)
                    generated_uuids.append(str(file_uploaded.uuid))

        # 5) Callback to Next.js with the generated image UUIDs
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