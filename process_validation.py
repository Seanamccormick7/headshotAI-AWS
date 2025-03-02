#!/usr/bin/env python
# process_validation.py
import os
import argparse
import tempfile
import glob
from tqdm import tqdm

try:
    import pyuploadcare
    from pyuploadcare import Uploadcare
    # Configure Uploadcare keys (ensure these environment variables are set)
    pyuploadcare.conf.api_secret = os.environ.get("UPLOADCARE_SECRET_KEY", "")
    pyuploadcare.conf.api_public = os.environ.get("UPLOADCARE_PUBLIC_KEY", "")
    HAS_UPLOADCARE = True
except ImportError:
    print("Warning: pyuploadcare not found. Image uploading will be disabled.")
    HAS_UPLOADCARE = False

def process_validation_images(
    input_dir: str,
    output_file: str = None,
    upload_to_uploadcare: bool = True,
    epoch: int = None,
    model_path: str = None,
    prompt: str = None,
):
    """
    Process validation images from a directory:
    1. Optionally upload them to Uploadcare
    2. Save the list of UUIDs to a file
    """
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' does not exist!")
        return []
    
    # Find all image files
    image_files = glob.glob(os.path.join(input_dir, "*.png")) + \
                  glob.glob(os.path.join(input_dir, "*.jpg")) + \
                  glob.glob(os.path.join(input_dir, "*.jpeg"))
    
    if not image_files:
        print(f"No image files found in '{input_dir}'!")
        return []
    
    print(f"Found {len(image_files)} image files to process in {input_dir}")
    
    # Upload images to Uploadcare if requested
    generated_uuids = []
    if upload_to_uploadcare and HAS_UPLOADCARE:
        try:
            uploadcare = Uploadcare()
            
            for filepath in tqdm(image_files, desc="Uploading validation images"):
                try:
                    with open(filepath, "rb") as file_obj:
                        file_uploaded = uploadcare.upload(file_obj)
                    uuid = str(file_uploaded.uuid)
                    generated_uuids.append(uuid)
                    print(f"Uploaded {os.path.basename(filepath)} → UUID: {uuid}")
                except Exception as e:
                    print(f"Error uploading {filepath}: {str(e)}")
        except Exception as e:
            print(f"Error initializing Uploadcare: {str(e)}")
    
    # Save UUIDs to file if requested
    if output_file and generated_uuids:
        with open(output_file, "w") as f:
            for uuid in generated_uuids:
                f.write(f"{uuid}\n")
        print(f"Saved {len(generated_uuids)} UUIDs to '{output_file}'")
    
    # Return the list of UUIDs
    return generated_uuids

def main():
    parser = argparse.ArgumentParser(description="Process validation images from DreamBooth training")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing validation images")
    parser.add_argument("--output_file", type=str, help="File to save UUIDs to")
    parser.add_argument("--no_upload", action="store_true", help="Skip uploading to Uploadcare")
    parser.add_argument("--epoch", type=int, help="Current epoch number (for logging)")
    parser.add_argument("--model_path", type=str, help="Path to the model (not used, for compatibility)")
    parser.add_argument("--prompt", type=str, help="Prompt used for generation (not used, for compatibility)")
    parser.add_argument("--process_only", action="store_true", help="Process existing images only (not used, for compatibility)")
    
    args = parser.parse_args()
    
    generated_uuids = process_validation_images(
        input_dir=args.input_dir,
        output_file=args.output_file,
        upload_to_uploadcare=not args.no_upload,
        epoch=args.epoch,
        model_path=args.model_path,
        prompt=args.prompt,
    )
    
    print(f"Processed {len(generated_uuids)} validation images")
    
if __name__ == "__main__":
    main()