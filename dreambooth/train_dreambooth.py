import argparse
import logging
import math
import os
import hashlib
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Enable TF32 for better performance on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="DreamBooth fine-tuning script for SD 3.5")

    # Required
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained SD3.5 model or HF repo, e.g. 'stabilityai/stable-diffusion-3.5-medium'.",
    )

    # Optional model revisions / paths
    parser.add_argument("--pretrained_vae_name_or_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None,
                        help="Revision of the pretrained model from huggingface.co/models.")

    # Tokenizer / text encoder
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--train_text_encoder", action="store_true",
                        help="Train the text encoders (CLIP-L, CLIP-G, T5). Potentially large VRAM usage.")

    # Data / prompts
    parser.add_argument("--instance_data_dir", type=str, default=None,
                        help="Folder containing instance images.")
    parser.add_argument("--instance_prompt", type=str, default=None,
                        help="Prompt with identifier specifying the instance (e.g. 'photo of sks person').")
    parser.add_argument("--class_data_dir", type=str, default=None,
                        help="Folder containing class images for prior-preservation.")
    parser.add_argument("--class_prompt", type=str, default=None,
                        help="Prompt for class images, if prior-preservation is used.")

    # Sample saving (end-of-training)
    parser.add_argument("--save_sample_prompt", type=str, default=None,
                        help="If set, generate sample images at the end.")
    parser.add_argument("--save_sample_negative_prompt", type=str, default=None)
    parser.add_argument("--n_save_sample", type=int, default=4)
    parser.add_argument("--save_guidance_scale", type=float, default=7.5)
    parser.add_argument("--save_infer_steps", type=int, default=20)

    # Prior preservation
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)

    # Training parameters
    parser.add_argument("--output_dir", type=str, default="text-inversion-model")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no","fp16","bf16"])
    parser.add_argument("--not_cache_latents", action="store_true")
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args(input_args) if input_args else parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare instance (and optional class) images with prompts for fine-tuning SD3.5.
    """

    def __init__(
        self,
        instance_data_dir,
        instance_prompt,
        tokenizer,
        with_prior_preservation=False,
        class_data_dir=None,
        class_prompt=None,
        num_class_images=0,
        size=512,
        center_crop=False,
        hflip=False,
    ):
        super().__init__()
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        instance_dir = Path(instance_data_dir)
        self.instance_images_path = sorted(
            [p for p in instance_dir.iterdir() if p.is_file() and not p.name.endswith(".txt")]
        )

        self.class_images_path = []
        if self.with_prior_preservation and class_data_dir:
            class_dir = Path(class_data_dir)
            self.class_images_path = sorted(
                [p for p in class_dir.iterdir() if p.is_file() and not p.name.endswith(".txt")]
            )
            if len(self.class_images_path) > num_class_images:
                self.class_images_path = self.class_images_path[:num_class_images]

        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)

        self._length = max(self.num_instance_images, self.num_class_images if self.with_prior_preservation else 0)
        if self._length == 0:
            self._length = self.num_instance_images

        self.image_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5 if hflip else 0.0),
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]),
        ])

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        example = {}

        inst_img_path = self.instance_images_path[idx % self.num_instance_images]
        inst_img = Image.open(inst_img_path).convert("RGB")
        example["instance_images"] = self.image_transforms(inst_img)

        # Tokenize instance prompt
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        # If prior-preservation
        if self.with_prior_preservation and self.class_images_path:
            cls_img_path = self.class_images_path[idx % self.num_class_images]
            cls_img = Image.open(cls_img_path).convert("RGB")
            example["class_images"] = self.image_transforms(cls_img)

            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    """
    For generating class images if not enough exist.
    """

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        return {"prompt": self.prompt, "index": i}


class LatentsDataset(Dataset):
    """
    For cached latents. We store a DiagonalGaussianDistribution for each image, plus text encoder data.
    """

    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, idx):
        # Each item is (latent_dist, text_data)
        latent_dist = self.latents_cache[idx]  # DiagonalGaussianDistribution
        latents = latent_dist.sample() * 0.18215
        return (latents, self.text_encoder_cache[idx])


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.seed is not None:
        set_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(args.output_dir, "logs"),
    )

    logger.info(args)

    # 1) Optionally generate class images if prior-preservation is enabled
    if args.with_prior_preservation and args.class_data_dir and args.class_prompt:
        class_dir = Path(args.class_data_dir)
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(class_dir.iterdir()))
        needed = args.num_class_images - existing
        if needed > 0:
            logger.info(f"Generating {needed} class images for prior-preservation.")
            pipe_prior = StableDiffusion3Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch.float16 if accelerator.device.type=="cuda" else torch.float32,
                safety_checker=None,
                revision=args.revision,
                use_auth_token=os.environ.get("HF_TOKEN"),
            )
            pipe_prior.scheduler = DDIMScheduler.from_config(pipe_prior.scheduler.config)
            if is_xformers_available():
                pipe_prior.enable_xformers_memory_efficient_attention()
            pipe_prior.to(accelerator.device)

            ds = PromptDataset(args.class_prompt, needed)
            dl = torch.utils.data.DataLoader(ds, batch_size=args.sample_batch_size)
            dl = accelerator.prepare(dl)
            with torch.autocast("cuda"), torch.inference_mode():
                for batch in tqdm(dl, desc="Generating class images"):
                    images = pipe_prior(batch["prompt"]).images
                    for i, img in enumerate(images):
                        name = f"{batch['index'][i] + existing}.jpg"
                        img.save(class_dir / name)

            del pipe_prior
            torch.cuda.empty_cache()

    # 2) Load pipeline, freeze or train text encoders, freeze vae
    logger.info("Loading StableDiffusion3Pipeline in float32 for training...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder_3=None,            # drop the T5-XXL
        tokenizer_3=None,              # drop the T5 tokenizer
        torch_dtype=torch.float32,
        use_auth_token=os.environ.get("HF_TOKEN"),
        safety_checker=None,
        revision=args.revision
    )

    vae = pipe.vae
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    text_encoder_3 = pipe.text_encoder_3  # newly added line
    transformer = pipe.transformer
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    tokenizer_3 = pipe.tokenizer_3  # newly added line

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        if text_encoder:   text_encoder.requires_grad_(False)
        if text_encoder_2: text_encoder_2.requires_grad_(False)
        if text_encoder_3: text_encoder_3.requires_grad_(False)

    if is_xformers_available():
        vae.enable_xformers_memory_efficient_attention()
        transformer.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            if text_encoder:   text_encoder.gradient_checkpointing_enable()
            if text_encoder_2: text_encoder_2.gradient_checkpointing_enable()
            if text_encoder_3: text_encoder_3.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Setup optimizer
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.train_text_encoder:
        params_to_optimize = (
            list(transformer.parameters()) +
            list(text_encoder.parameters()) +
            list(text_encoder_2.parameters()) +
            list(text_encoder_3.parameters())  # Add T5 encoder
        )
    else:
        params_to_optimize = transformer.parameters()

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 3) Create dataset + dataloader
    train_dataset = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,  # main tokenizer
        with_prior_preservation=args.with_prior_preservation,
        class_data_dir=args.class_data_dir,
        class_prompt=args.class_prompt,
        num_class_images=args.num_class_images,
        size=args.resolution,
        center_crop=args.center_crop,
        hflip=args.hflip,
    )

    def collate_fn(examples):
        pixel_values = []
        input_ids_list = []
        for ex in examples:
            pixel_values.append(ex["instance_images"])
            input_ids_list.append(ex["instance_prompt_ids"])

        if args.with_prior_preservation and args.class_data_dir and args.class_prompt:
            pixel_values += [ex["class_images"] for ex in examples]
            input_ids_list += [ex["class_prompt_ids"] for ex in examples]

        pixel_values = torch.stack(pixel_values).float().contiguous()
        input_ids = tokenizer.pad({"input_ids": input_ids_list}, padding=True, return_tensors="pt").input_ids

        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Move VAE + text_encoders to GPU if not caching latents
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        if text_encoder:   text_encoder.to(accelerator.device, dtype=weight_dtype)
        if text_encoder_2: text_encoder_2.to(accelerator.device, dtype=weight_dtype)
        if text_encoder_3: text_encoder_3.to(accelerator.device, dtype=weight_dtype)

    # 4) Optionally cache latents
    if not args.not_cache_latents:
        logger.info("Caching latents for dataset...")
        latents_cache = []
        text_encoder_cache = []

        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                pv = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                ids = batch["input_ids"].to(accelerator.device)
                latent_dist = vae.encode(pv).latent_dist
                latents_cache.append(latent_dist)

                if args.train_text_encoder:
                    text_encoder_cache.append(ids)
                else:
                    # we call pipeline.encode_prompt(...) if we wanted the multi-encoder aggregator
                    # but for simplicity, let's call text_encoder(...) for single CLIP. Or do the aggregator if needed.
                    # If you want the aggregator, do:
                    prompt_embeds = pipe.encode_prompt(
                        ids,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        max_sequence_length=256  # for T5
                    )
                    text_encoder_cache.append(prompt_embeds)

        from torch.utils.data import DataLoader
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        # free memory
        del vae
        if not args.train_text_encoder:
            if text_encoder:   del text_encoder
            if text_encoder_2: del text_encoder_2
            if text_encoder_3: del text_encoder_3
        torch.cuda.empty_cache()

    # 5) Scheduler logic
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps*args.gradient_accumulation_steps,
    )

    # 6) Prepare models + data with accelerator
    objs_to_prepare = [transformer, optimizer, train_dataloader, lr_scheduler]
    if args.train_text_encoder:
        objs_to_prepare = [transformer, text_encoder, text_encoder_2, text_encoder_3, optimizer, train_dataloader, lr_scheduler]

    prepared = accelerator.prepare(*objs_to_prepare)
    idx = 0
    if args.train_text_encoder:
        transformer = prepared[idx]; idx+=1
        text_encoder = prepared[idx]; idx+=1
        text_encoder_2 = prepared[idx]; idx+=1
        text_encoder_3 = prepared[idx]; idx+=1
        optimizer = prepared[idx]; idx+=1
        train_dataloader = prepared[idx]; idx+=1
        lr_scheduler = prepared[idx]; idx+=1
    else:
        transformer = prepared[idx]; idx+=1
        optimizer = prepared[idx]; idx+=1
        train_dataloader = prepared[idx]; idx+=1
        lr_scheduler = prepared[idx]; idx+=1

    accelerator.init_trackers("dreambooth_sd3.5")

    # 7) Training loop
    global_step = 0
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # If not training text encoder, we won't do backprop for them
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()

    for epoch in range(9999999):
        transformer.train()
        if args.train_text_encoder:
            if text_encoder:   text_encoder.train()
            if text_encoder_2: text_encoder_2.train()
            if text_encoder_3: text_encoder_3.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # If latents are cached, batch = [ latents, text_data ]
                # If not cached, batch = { "pixel_values", "input_ids" }
                if not args.not_cache_latents:
                    latents, text_data = batch
                    bsz = latents.shape[0]
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    with text_enc_context:
                        if args.train_text_encoder:
                            # text_data is input_ids
                            # we call pipe.encode_prompt with the aggregator?
                            # Or do text_encoder(...) if you only want single-CLIP.
                            # We'll do aggregator for multi-encoder:
                            prompt_embeds = pipe.encode_prompt(
                                text_data,  # input_ids
                                device=latents.device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                max_sequence_length=256
                            )
                            encoder_hidden_states = prompt_embeds
                        else:
                            # text_data is already "prompt_embeds" if we stored them
                            encoder_hidden_states = text_data

                else:
                    # not caching latents
                    pv = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                    ids = batch["input_ids"].to(accelerator.device)
                    latent_dist = vae.encode(pv).latent_dist
                    latents = latent_dist.sample() * 0.18215

                    bsz = latents.shape[0]
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    with text_enc_context:
                        if args.train_text_encoder:
                            # aggregator
                            prompt_embeds = pipe.encode_prompt(
                                ids,
                                device=latents.device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                max_sequence_length=256
                            )
                            encoder_hidden_states = prompt_embeds
                        else:
                            # aggregator but text enc. is frozen
                            encoder_hidden_states = pipe.encode_prompt(
                                ids,
                                device=latents.device,
                                num_images_per_prompt=1,
                                do_classifier_free_guidance=False,
                                max_sequence_length=256
                            )

                # Forward pass in the "transformer"
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # MSE or V-pred
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError("Unknown prediction_type in noise_scheduler")

                if args.with_prior_preservation and args.class_data_dir and args.class_prompt:
                    half = bsz // 2
                    mp1, mp2 = model_pred[:half], model_pred[half:]
                    tar1, tar2 = target[:half], target[half:]
                    loss = F.mse_loss(mp1.float(), tar1.float(), reduction="mean")
                    prior_loss = F.mse_loss(mp2.float(), tar2.float(), reduction="mean")
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % 10 == 0:
                progress_bar.set_postfix({"loss": loss.item()})

            progress_bar.update(1)
            global_step += 1
            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    # 8) Final Save
    if accelerator.is_main_process:
        logger.info("** Saving final pipeline weights **")

        # Unwrap
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        pipe.transformer = unwrapped_transformer

        if args.train_text_encoder:
            pipe.text_encoder = accelerator.unwrap_model(pipe.text_encoder)
            pipe.text_encoder_2 = accelerator.unwrap_model(pipe.text_encoder_2)
            pipe.text_encoder_3 = accelerator.unwrap_model(pipe.text_encoder_3)

        pipe.to(dtype=torch.float16)
        pipe.save_pretrained(args.output_dir)
        logger.info(f"Model saved at {args.output_dir}")

        # Optionally sample final images
        if args.save_sample_prompt:
            logger.info("Generating sample images with final pipeline...")
            pipe.to(accelerator.device)
            g_cuda = torch.Generator(device=accelerator.device)
            if args.seed is not None:
                g_cuda.manual_seed(args.seed)

            sample_dir = os.path.join(args.output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            with torch.autocast("cuda"), torch.inference_mode():
                for i in tqdm(range(args.n_save_sample), desc="Generating samples"):
                    images = pipe(
                        prompt=args.save_sample_prompt,
                        negative_prompt=args.save_sample_negative_prompt,
                        guidance_scale=args.save_guidance_scale,
                        num_inference_steps=args.save_infer_steps,
                        generator=g_cuda,
                    ).images
                    images[0].save(os.path.join(sample_dir, f"sample_{i}.png"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
