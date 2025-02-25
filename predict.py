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
    validation_prompt: str = None,
    sample_negative_prompt: str = "",
    n_save_sample: int = 2,
    save_guidance_scale: float = 7.5,
    save_infer_steps: int = 20,
    class_prompt: str = None,  # Added class prompt parameter
    class_data_dir: str = None  # Added class data directory parameter
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
        "dreambooth/train_dreambooth_lora_sd3.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-3-medium-diffusers",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--resolution=512",
        "--train_batch_size=1",
        "--train_text_encoder",
        "--mixed_precision=fp16",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps=4",
        "--learning_rate=4e-4",
        "--lr_scheduler=constant",
        "--seed=40",
        "--report_to=tensorboard",
        f"--max_train_steps={steps}",
        f"--output_dir={output_dir}",
        f"--instance_data_dir={instance_data_dir}",
        f"--instance_prompt={instance_prompt}",
        f"--class_data_dir={class_data_dir}",
        f"--class_prompt={class_prompt}",
        f"--validation_prompt={validation_prompt}",
        "--num_validation_images=20",
        "--validation_epochs=1",
    ]

    subprocess.run(cmd, check=True)
    return f"Training complete. Model saved at {output_dir}/"