# inference.py
import os
import torch
from diffusers import StableDiffusion3Pipeline, DDIMScheduler, DPMSolverMultistepScheduler

def run_inference(
    base_model="stabilityai/stable-diffusion-3-medium-diffusers",
    lora_weights_path="trained_model/pytorch_lora_weights.safetensors",
    prompt="Professional headshot photo of a man with brown hair, 30-year-old, wearing a suit, in a studio setting.", 
    negative_prompt="blurry, deformed, bad anatomy, disfigured, poorly drawn face, distorted face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, out of focus",
    outdir="inference_output",
    num_images=5, #will change to 100 in production
    guidance_scale=7.5,
    num_inference_steps=30,
    torch_dtype=torch.float16,
):
    """
    Loads base SD3 model, applies LoRA weights, generates images from `prompt`,
    and saves them into `outdir` directory.
    """
    os.makedirs(outdir, exist_ok=True)

    # 1) Load the base SD3 pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype
    ).to("cuda")
    #pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)   # or can set inference steps to 100
    # 2) Load your LoRA weights
    pipe.load_lora_weights(lora_weights_path)

    # 3) Generate images in a loop
    for i in range(num_images):
        with torch.autocast("cuda"):
            image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, negative_prompt=negative_prompt).images[0]
        save_path = os.path.join(outdir, f"sample_{i}.png")
        image.save(save_path)

    print(f"Inference complete! {num_images} images saved to {outdir}")
    return outdir
