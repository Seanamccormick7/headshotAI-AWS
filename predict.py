# predict.py
import os
import subprocess
import glob
import shutil

def run_training(
    instance_data: str,
    instance_prompt: str,
    steps: int = 10, #default value (can actually change in tasks.py)
    output_dir: str = "trained_model",
    class_prompt: str = None,  # Added class prompt parameter
    class_data_dir: str = None,  # Added class data directory parameter
    num_class_images: int = 50,  # Added number of class images parameter
) -> str:
    """Main training function without Cog dependencies."""

    # Prepare / extract instance data
    instance_data_dir = "/src/instance_images"
    if os.path.isdir(instance_data):
        instance_data_dir = str(instance_data)
    else:
        os.makedirs(instance_data_dir, exist_ok=True)
        # Just unzip directly, no moving subfolders
        subprocess.run(["unzip", "-o", instance_data, "-d", instance_data_dir], check=True)

    # Build the training command
    cmd = [
        "accelerate", "launch",
        "dreambooth/train_dreambooth_sd3.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-3-medium-diffusers",
        "--with_prior_preservation",
        "--prior_loss_weight=0.5",
        "--resolution=512",         #can change to 768 for better resolution
        "--train_batch_size=1",     
        "--gradient_accumulation_steps=2",      #can change to 4 if needed
        #"--train_text_encoder",         #not working for now
        "--mixed_precision=fp16",
        "--use_8bit_adam",              #to reduce memory usage
        "--gradient_checkpointing",
        "--learning_rate=5e-6",     #or 2e-6 if not working well
        "--lr_scheduler=constant",
        "--seed=40",
        "--report_to=tensorboard",
        f"--max_train_steps={steps}",    #trying 800, but can increase to 1200 if possible
        f"--output_dir={output_dir}",
        f"--instance_data_dir={instance_data_dir}",
        f"--instance_prompt={instance_prompt}",
        f"--class_data_dir={class_data_dir}",
        f"--class_prompt={class_prompt}",
        f"--num_class_images={num_class_images}",
    ]

    subprocess.run(cmd, check=True)
    return f"Training complete. Model saved at {output_dir}/"
