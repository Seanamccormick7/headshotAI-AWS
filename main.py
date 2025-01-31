import os
import uvicorn
import requests
import tempfile
import zipfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from predict import run_training
import pyuploadcare

# You may have to set your Uploadcare keys in environment or directly here
pyuploadcare.conf.api_secret = os.environ.get("UPLOADCARE_SECRET_KEY", "secret")
pyuploadcare.conf.api_public = os.environ.get("UPLOADCARE_PUBLIC_KEY", "public")

app = FastAPI()

class GenerateRequest(BaseModel):
    userId: str
    gender: str = None
    hairColor: str = None
    hairLength: str = None
    ethnicity: str = None
    bodyType: str = None
    attire: str = None
    backgrounds: str = None
    glasses: bool = False
    instanceImages: List[str] = []  # array of Uploadcare UUIDs
    callbackUrl: str

@app.post("/generate")
def generate_images(req: GenerateRequest):
    """
    1) Download user instance images from Uploadcare
    2) Run DreamBooth training + sampling (via your predictor or direct script)
    3) Upload resulting images to Uploadcare, get back a list of UUIDs
    4) POST those UUIDs to Next.js callbackUrl
    """
    try:
        # 1) Download user’s instance images from Uploadcare into a local folder or zip
        #    Below is simplistic example. You'll likely want them as separate .jpg files.

        instance_dir = tempfile.mkdtemp()
        for i, uuid in enumerate(req.instanceImages):
            download_url = f"https://ucarecdn.com/{uuid}/-/format/jpeg/"
            # Save to {instance_dir}/{i}.jpg
            out_path = os.path.join(instance_dir, f"{i}.jpg")
            
            import requests
            r = requests.get(download_url)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
            else:
                print(f"Failed to download {uuid}, status {r.status_code}")

        # 2) Run DreamBooth fine-tuning using your `Predictor` from predict.py
        #    Typically you pass the instance images as a zip or folder. We'll zip them:

        zip_path = os.path.join(instance_dir, "images.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for img_file in os.listdir(instance_dir):
                if img_file.endswith(".jpg"):
                    zf.write(os.path.join(instance_dir, img_file), arcname=img_file)

        # For example, you pass in instance_data, instance_prompt, steps, etc.
        # We'll supply a fake instance_prompt. 
        # In reality, you might build a more descriptive prompt from the user’s attributes.
        instance_prompt = "photo of sks person" #with certain details
        training_steps = 10 #increase when real training

        # Run the training (this can take up to 2+ hours)
        train_output = run_training(
            instance_data=zip_path,
            instance_prompt=instance_prompt,
            steps=training_steps,
            sample_prompt="high quality professional headshot of sks person"
        )
        print("DreamBooth output:", train_output)

        # 3) Suppose you produce 150 images. For example, let's say your code stored them in some folder:
        #    You would want to find all .png/.jpg in that folder and upload them to Uploadcare:

        generated_uuids = []
        output_images_dir = "trained_model/10/samples"  # needs to match steps above

        if not os.path.isdir(output_images_dir):
            print("No images found. (check your script output)")
        else:
            from pyuploadcare import File
            from pyuploadcare import Uploadcare
            uploadcare = Uploadcare()

            for filename in os.listdir(output_images_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    fullpath = os.path.join(output_images_dir, filename)
                    # Upload:
                    fileObj = uploadcare.upload(open(fullpath, "rb"))
                    # The returned fileObj has a .uuid property
                    generated_uuids.append(str(fileObj.uuid))

        # 4) Callback to Next.js
        callback_payload = {
            "userId": req.userId,
            "generatedUuids": generated_uuids  # array of 150 new image UUIDs
        }
        resp = requests.post(req.callbackUrl, json=callback_payload)
        if not resp.ok:
            print(f"Callback to Next.js failed: {resp.text}")

        return {"message": "Generation started successfully. Callback will be or has been sent."}

    except Exception as e:
        print("Error in generate_images:", e)
        raise HTTPException(status_code=500, detail=str(e))

# If you want to run locally (not strictly needed for Cog, but useful for testing):
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
