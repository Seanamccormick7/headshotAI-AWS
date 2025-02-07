# predict.py
import os
import subprocess
import glob
import shutil

def run_training(
    instance_data: str,
    instance_prompt: str,
    steps: int = 800,
    output_dir: str = "trained_model",
    sample_prompt: str = "",
    sample_negative_prompt: str = "",
    n_save_sample: int = 2,
    save_guidance_scale: float = 7.5,
    save_infer_steps: int = 20
) -> str:
    """Main training function without Cog dependencies"""
    
    # Existing data handling logic
    instance_data_dir = "/src/instance_images"
    if os.path.isdir(instance_data):
        instance_data_dir = str(instance_data)
    else:
        # Ensure the extraction directory exists
        os.makedirs(instance_data_dir, exist_ok=True)
        subprocess.run(["unzip", "-o", instance_data, "-d", instance_data_dir], check=True)
        for folder in glob.glob(os.path.join(instance_data_dir, "*/")):
            subprocess.run(["mv", f"{folder}*", instance_data_dir], check=True)
            subprocess.run(["rm", "-r", folder], check=True)
        instance_data_dir = "/src/instance_images"

    # Updated command with memory optimization flags
    cmd = [
        "python", "dreambooth/train_dreambooth.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-3.5-medium",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--seed=42",
        "--resolution=512",
        "--train_batch_size=1",
        "--train_text_encoder",
        "--mixed_precision=fp16",
        "--use_8bit_adam",
        "--gradient_checkpointing",  # Added for memory efficiency
        "--gradient_accumulation_steps=4",  # Increased from 1
        "--learning_rate=1e-6",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--num_class_images=25",  # Reduced from 50
        "--sample_batch_size=1",  # Reduced from 4
        "--max_train_steps=10",
        f"--output_dir={output_dir}",
        "--save_interval=400",
        "--concepts_list=dreambooth/concepts_list.json",
        f"--instance_data_dir={instance_data_dir}",
        f"--instance_prompt={instance_prompt}",
        "--enable_xformers_memory_efficient_attention",  # Added for memory efficiency
    ]

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