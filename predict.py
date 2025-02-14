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
    sample_prompt: str = "",
    sample_negative_prompt: str = "",
    n_save_sample: int = 2,
    save_guidance_scale: float = 7.5,
    save_infer_steps: int = 20
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
        "python", "dreambooth/train_dreambooth.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-3.5-medium",
        "--with_prior_preservation",              # Keep if you want prior-preservation
        "--prior_loss_weight=1.0",
        "--seed=42",
        "--resolution=512",
        "--train_batch_size=1", #"--train_text_encoder", add to train text encoder (high memory usage)
        "--mixed_precision=bf16",
        "--use_8bit_adam",
        "--gradient_checkpointing",
        "--gradient_accumulation_steps=1",
        "--learning_rate=1e-6",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--num_class_images=15",  # for prior-preservation
        "--sample_batch_size=1",
        f"--max_train_steps={steps}",
        f"--output_dir={output_dir}",
        f"--instance_data_dir={instance_data_dir}",
        f"--instance_prompt={instance_prompt}",
        "--train_batch_size=1",
    ]

    # If we want to save sample images after training, pass the sample args
    if sample_prompt:
        cmd += [
            f"--save_sample_prompt={sample_prompt}",
            f"--n_save_sample={n_save_sample}",
            f"--save_guidance_scale={save_guidance_scale}",
            f"--save_infer_steps={save_infer_steps}",
        ]
        if sample_negative_prompt:
            cmd.append(f"--save_sample_negative_prompt={sample_negative_prompt}")

    subprocess.run(cmd, check=True)
    return f"Training complete. Model saved at {output_dir}/"
