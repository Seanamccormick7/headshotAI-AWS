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

from celery_app import celery_app
from predict import run_training
from inference import run_inference

# Configure Uploadcare keys (ensure these environment variables are set)
pyuploadcare.conf.api_secret = os.environ.get("UPLOADCARE_SECRET_KEY", "secret")
pyuploadcare.conf.api_public = os.environ.get("UPLOADCARE_PUBLIC_KEY", "public")

@celery_app.task(bind=True)
def generate_images_task(self, req_data: dict):
    """
    Celery task to process image generation.
    Downloads user images, runs training, uploads the generated images,
    calls back to Next.js, and then cleans up temporary data.
    """
    instance_dir = None
    output_dir = None
    try:
        userId = req_data.get("userId")
        instanceImages = req_data.get("instanceImages", [])
        callbackUrl = req_data.get("callbackUrl")

        # 1) Download instance images
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

        # class_prompt used for prior preservation
        class_prompt = "Photo of a person"
        # instance prompt Used to learn the specific subject (the user-uploaded photos).
        instance_prompt = (
            f"Photo of a sks {age_str}{gender} with {hairColor} hair "
            f"({hairLength}), of {ethnicity} ethnicity, {bodyType} build, {glasses_str}."
        )

        # The final prompt for generating images after training
        inference_prompt = (
            f"Professional studio headshot of the sks {age_str}{gender} with {hairColor} hair "
            f"({hairLength}), {ethnicity} ethnicity, {bodyType} build, {glasses_str}, "
            f"wearing {attire}, in {backgrounds}, shot with professional lighting, "
            f"high detail, 4k, sharp focus, DSLR, professional portrait photography, "
            f"high-end editorial photography, trending on artstation, highly detailed"
        )

        # 4) Set training steps
        training_steps = 1200

        # 5) Create output directory
        output_dir = tempfile.mkdtemp(prefix="trained_model_")

        # 6) DreamBooth training (LoRA)
        train_output = run_training(
            instance_data=zip_path,
            instance_prompt=instance_prompt,
            steps=training_steps,
            output_dir=output_dir,
            class_prompt=class_prompt,
            class_data_dir="class_images",
            num_class_images=50,        #can do more than are in class images file (program will just generate more) 
        )
        print("DreamBooth training output:", train_output)
        print("Instance prompt:", instance_prompt)
        print("Inference prompt:", inference_prompt)

        # 7) Post-training inference
        # We'll store these final images in: e.g. output_dir + "/inference"
        inference_dir = os.path.join(output_dir, "inference_output")
        run_inference(
            base_model="stabilityai/stable-diffusion-3-medium-diffusers",
            lora_weights_path=os.path.join(output_dir, "pytorch_lora_weights.safetensors"),
            prompt=inference_prompt,
            outdir=inference_dir,
            num_images=5,  # doing 5 just for testing, change to 100 in production
            guidance_scale=7.5,
            num_inference_steps=30,
            torch_dtype=torch.float16,
        )

        # 8) Upload generated images to Uploadcare
        generated_uuids = []
        if not os.path.isdir(inference_dir):
            print("No inference images found in", inference_dir)
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
            print(f"Callback to Next.js failed: {resp.text}")

        return {"message": "Generation task completed", "generatedUuids": generated_uuids}

    except Exception as e:
        print("Error in generate_images_task:", str(e))
        raise e
    finally:
        # Cleanup
        if instance_dir and os.path.exists(instance_dir):
            shutil.rmtree(instance_dir, ignore_errors=True)
        if output_dir and os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)