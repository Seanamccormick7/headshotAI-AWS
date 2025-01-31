# predict.py
import subprocess
import glob
from pathlib import Path as PlyPath
import os
import sys
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def predict(
        self,
        instance_data: Path = Input(
            description="Zip or directory with instance images",
        ),
        instance_prompt: str = Input(
            description="Prompt for the instance, e.g., 'photo of sks person'"
        ),
        steps: int = Input(
            description="Number of training steps",
            default=800
        ),
        output_dir: Path = Input(
            description="Where to save model weights",
            default="trained_model"
        ),
        # --- New inputs for in-line sample generation ---
        sample_prompt: str = Input(
            description="Prompt used to generate sample images after training. Leave blank to skip sample generation.",
            default=""
        ),
        sample_negative_prompt: str = Input(
            description="Negative prompt for sample generation (optional).",
            default=""
        ),
        n_save_sample: int = Input(
            description="Number of sample images to save.",
            default=2
        ),
        save_guidance_scale: float = Input(
            description="CFG (guidance scale) for sample generation.",
            default=7.5
        ),
        save_infer_steps: int = Input(
            description="Number of diffusion steps for sample generation.",
            default=20
        ),
    ) -> str:
        """
        Train DreamBooth. Optionally generate sample images at the end.
        Return path to final model directory.
        """

        # Debugging: Print parameter values and types
        print(f"instance_data: {instance_data} (type: {type(instance_data)})", file=sys.stderr)
        print(f"instance_prompt: {instance_prompt} (type: {type(instance_prompt)})", file=sys.stderr)
        print(f"steps: {steps} (type: {type(steps)})", file=sys.stderr)
        print(f"output_dir: {output_dir} (type: {type(output_dir)})", file=sys.stderr)
        print(f"sample_prompt: {sample_prompt} (type: {type(sample_prompt)})", file=sys.stderr)
        print(f"sample_negative_prompt: {sample_negative_prompt} (type: {type(sample_negative_prompt)})", file=sys.stderr)
        print(f"n_save_sample: {n_save_sample} (type: {type(n_save_sample)})", file=sys.stderr)
        print(f"save_guidance_scale: {save_guidance_scale} (type: {type(save_guidance_scale)})", file=sys.stderr)
        print(f"save_infer_steps: {save_infer_steps} (type: {type(save_infer_steps)})", file=sys.stderr)

        # 1. Unzip or ensure instance_data is accessible
        instance_data_dir = PlyPath("/src/instance_images")
        if instance_data.is_dir():
            # If the user provided a directory
            instance_data_dir = instance_data
        else:
            # Ensure the extraction directory exists
            instance_data_dir.mkdir(parents=True, exist_ok=True)

            # If the user uploaded a zip, unzip it
            try:
                subprocess.run(["unzip", "-o", str(instance_data), "-d", str(instance_data_dir)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Failed to unzip {instance_data}: {e}", file=sys.stderr)
                raise

            # Flatten subfolders
            for folder in instance_data_dir.glob("*/"):
                try:
                    subprocess.run(["mv", f"{folder}*", str(instance_data_dir)], check=True)
                    subprocess.run(["rm", "-r", str(folder)], check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to flatten folder {folder}: {e}", file=sys.stderr)
                    raise

        # 2. Build the command for train_dreambooth.py
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

        # 2A. Append sample-generation flags if sample_prompt is non-empty
        if sample_prompt:
            cmd += [
                f"--save_sample_prompt={sample_prompt}",
                f"--n_save_sample={n_save_sample}",
                f"--save_guidance_scale={save_guidance_scale}",
                f"--save_infer_steps={save_infer_steps}",
            ]
            if sample_negative_prompt:
                cmd.append(f"--save_sample_negative_prompt={sample_negative_prompt}")

        # Debugging: Print the constructed command
        print(f"Constructed command: {' '.join(cmd)}", file=sys.stderr)

        # 3. Run the training
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training script failed with error: {e}", file=sys.stderr)
            raise

        # 4. Return the path to the model directory
        return f"Training complete. Model saved at {output_dir}/"

