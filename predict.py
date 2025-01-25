# predict.py
from cog import BasePredictor, Input, Path
import os
import subprocess
import glob

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
        output_dir: str = Input(
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

        # 1. Unzip or ensure instance_data is accessible
        instance_data_dir = "/src/instance_images"
        if instance_data.is_dir():
            # If the user provided a directory
            instance_data_dir = str(instance_data)
        else:
            # If the user uploaded a zip, unzip it
            os.system(f"unzip -o {instance_data} -d /src/instance_images")

            # Flatten subfolders
            for folder in glob.glob("/src/instance_images/*/"):
                os.system(f"mv {folder}* /src/instance_images/")
                os.system(f"rm -r {folder}")

            instance_data_dir = "/src/instance_images"

        # 2. Build the command for train_dreambooth.py
        cmd = [
            "python", "dreambooth/train_dreambooth.py",
            # Matches your old script
            "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
            "--pretrained_vae_name_or_path=stabilityai/sd-vae-ft-mse",
            "--with_prior_preservation",
            "--prior_loss_weight=1.0",
            "--seed=3434554",  # override the old 42
            "--resolution=512",
            "--train_batch_size=1",
            "--train_text_encoder",
            "--mixed_precision=fp16",
            "--use_8bit_adam",
            "--gradient_accumulation_steps=1",
            "--learning_rate=1e-6",  # override your old 5e-6
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
            "--num_class_images=50",
            "--sample_batch_size=4",
            # We'll keep your dynamic:
            f"--max_train_steps={steps}",
            f"--output_dir={output_dir}",
            "--save_interval=400",
            # Provide your concepts_list, if you want to use that approach
            "--concepts_list=dreambooth/concepts_list.json",
            # instance data folder, prompt
            f"--instance_data_dir={instance_data_dir}",
            f"--instance_prompt={instance_prompt}",
        ] 

        # 2A. Append sample-generation flags if sample_prompt is non-empty
        #     (You can also check for sample_negative_prompt, etc.)
        if sample_prompt:
            cmd += [
                f"--save_sample_prompt={sample_prompt}",
                f"--n_save_sample={n_save_sample}",
                f"--save_guidance_scale={save_guidance_scale}",
                f"--save_infer_steps={save_infer_steps}",
            ]
            if sample_negative_prompt:
                cmd.append(f"--save_sample_negative_prompt={sample_negative_prompt}")

        # 3. Run the training
        subprocess.run(cmd, check=True)

        # 4. Return the path to the model directory
        return f"Training complete. Model saved at {output_dir}/"
