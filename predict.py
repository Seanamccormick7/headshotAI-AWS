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
        os.system(f"unzip -o {instance_data} -d /src/instance_images")
        for folder in glob.glob("/src/instance_images/*/"):
            os.system(f"mv {folder}* /src/instance_images/")
            os.system(f"rm -r {folder}")
        instance_data_dir = "/src/instance_images"

    # Build command (same as before)
    cmd = [
        "python", "dreambooth/train_dreambooth.py",
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse",
        "--with_prior_preservation",
        "--prior_loss_weight=1.0",
        "--seed=42",
        "--resolution=512",
        "--train_batch_size=1",
        "--train_text_encoder",
        "--mixed_precision=fp16",
        "--use_8bit_adam",
        "--gradient_accumulation_steps=1",
        "--learning_rate=1e-6",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--num_class_images=50",
        "--sample_batch_size=4",
        f"--max_train_steps={steps}",
        f"--output_dir={output_dir}",
        "--save_interval=400",
        "--concepts_list=dreambooth/concepts_list.json",
        f"--instance_data_dir={instance_data_dir}",
        f"--instance_prompt={instance_prompt}",
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
