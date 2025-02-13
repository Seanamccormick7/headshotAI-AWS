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
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    # Optional model revisions / paths
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="(Unused) in SD3.5 - the model includes its own VAE by default.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    # Tokenizer / text encoder
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder",
    )

    # Data / prompts
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the class images for prior preservation.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class, used for prior preservation.",
    )

    # Sample saving (end-of-training)
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="If specified, generate sample images using this prompt after training.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="Negative prompt for the sample generation.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="Number of samples to save at the end of training.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG scale for sample generation.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=20,
        help="Number of inference steps for sample generation.",
    )

    # Prior preservation
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help="Minimal class images for prior preservation loss.",
    )

    # Training parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution for input images.",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling class images or validation images.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory.",
    )

    # Optimizer / LR
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='Type of scheduler: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for warmup in the lr scheduler.",
    )
    parser.add_argument("--use_8bit_adam", action="store_true", help="Use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Adam epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max grad norm.")

    # Mixed precision
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Use mixed precision. Choose between fp16 or bf16 (requires Ampere GPU).",
    )

    # Caching
    parser.add_argument(
        "--not_cache_latents",
        action="store_true",
        help="Do not precompute and cache latents from VAE.",
    )

    # Data augmentation
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")

    # Internal/distributed
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.")

    args = parser.parse_args(input_args) if input_args else parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance (and optional class) images with the prompts for fine-tuning.
    It pre-processes the images and tokenizes prompts.
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

        # Collect instance images
        instance_dir = Path(instance_data_dir)
        self.instance_images_path = sorted(
            [p for p in instance_dir.iterdir() if p.is_file() and not p.name.endswith(".txt")]
        )

        # Collect class images if prior preservation is used
        self.class_images_path = []
        if self.with_prior_preservation and class_data_dir:
            class_dir = Path(class_data_dir)
            # You may want to limit to num_class_images in practice
            self.class_images_path = sorted(
                [p for p in class_dir.iterdir() if p.is_file() and not p.name.endswith(".txt")]
            )
            # If user wants to limit to "num_class_images", do so:
            if len(self.class_images_path) > num_class_images:
                self.class_images_path = self.class_images_path[:num_class_images]

        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)

        # The dataset length is the max of the two so we can sample instance + class in each batch
        self._length = max(self.num_instance_images, self.num_class_images if self.with_prior_preservation else 0)
        if self._length == 0:
            self._length = self.num_instance_images  # fallback if no class images

        self.image_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5 if hflip else 0.0),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        # Instance
        instance_image_path = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image_path)
        if instance_image.mode != "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        # Class
        if self.with_prior_preservation and len(self.class_images_path) > 0:
            class_image_path = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_image_path)
            if class_image.mode != "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    """A simple dataset to generate class images for prior preservation (if needed)."""

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"prompt": self.prompt, "index": idx}


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        # Sample from the DiagonalGaussianDistribution
        latent = self.latents_cache[index].sample() * 0.18215
        return (latent, self.text_encoder_cache[index])


def main(args):
    logging_dir = os.path.join(args.output_dir, "logs")

    huggingface_token = os.environ.get("HF_TOKEN")
    if huggingface_token is None:
        logger.warning("No HF_TOKEN found in environment. Gated models may fail (401).")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # -------------------------------------------------------------------------
    # 1. (Optional) Generate class images for prior preservation
    # -------------------------------------------------------------------------
    if args.with_prior_preservation and args.class_data_dir and args.class_prompt:
        class_images_dir = Path(args.class_data_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
            pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                revision=args.revision,
                use_auth_token=huggingface_token,
            )
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            if is_xformers_available():
                pipeline.enable_xformers_memory_efficient_attention()
            pipeline.to(accelerator.device)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Generating {num_new_images} class images for prior-preservation.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
            sample_dataloader = accelerator.prepare(sample_dataloader)

            with torch.autocast("cuda"), torch.inference_mode():
                for batch in tqdm(sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process):
                    images = pipeline(batch["prompt"], num_inference_steps=args.save_infer_steps).images
                    for i, image in enumerate(images):
                        # Save each image uniquely
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        filename = class_images_dir / f"{batch['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 2. Load tokenizer, text_encoder, VAE, transformer
    # -------------------------------------------------------------------------
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_auth_token=huggingface_token,
        )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        use_auth_token=huggingface_token,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        use_auth_token=huggingface_token,
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        torch_dtype=torch.float32,
        use_auth_token=huggingface_token,
    )

    vae.requires_grad_(False)
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if is_xformers_available():
        vae.enable_xformers_memory_efficient_attention()
        transformer.enable_xformers_memory_efficient_attention()
    else:
        logger.warning("xformers is not installed. For best performance, install it.")

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        # Scale the learning rate by number of GPUs, gradient accumulation steps, and batch size
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
        params_to_optimize = list(transformer.parameters()) + list(text_encoder.parameters())
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

    # -------------------------------------------------------------------------
    # 3. Create dataset and dataloader
    # -------------------------------------------------------------------------
    train_dataset = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        class_data_dir=args.class_data_dir,
        class_prompt=args.class_prompt,
        num_class_images=args.num_class_images,
        size=args.resolution,
        center_crop=args.center_crop,
        hflip=args.hflip,
    )

    def collate_fn(examples):
        # Gather instance images
        input_ids_list = [ex["instance_prompt_ids"] for ex in examples]
        pixel_values_list = [ex["instance_images"] for ex in examples]

        # If prior preservation, gather class images too
        if args.with_prior_preservation and args.class_data_dir and args.class_prompt:
            input_ids_list += [ex["class_prompt_ids"] for ex in examples]
            pixel_values_list += [ex["class_images"] for ex in examples]

        pixel_values = torch.stack(pixel_values_list)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad({"input_ids": input_ids_list}, padding=True, return_tensors="pt").input_ids

        batch_out = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch_out

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Determine weight dtype for GPU
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Optionally cache latents
    if not args.not_cache_latents:
        latents_cache = []
        text_encoder_cache = []

        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device)

                latent_dist = vae.encode(batch["pixel_values"]).latent_dist
                latents_cache.append(latent_dist)

                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(text_encoder(batch["input_ids"])[0])

        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

        # Free some memory
        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None or args.max_train_steps == 0:
        args.max_train_steps = num_update_steps_per_epoch
        overrode_max_train_steps = True

    # LR scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models/dataloaders
    if args.train_text_encoder:
        transformer, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = num_update_steps_per_epoch

    # For logging / progress
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (parallel * accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0

    # If we are not training the text_encoder, we won't compute gradients for it
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()

    # -------------------------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------------------------
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(args.num_train_epochs):
        transformer.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Convert images to latents (if not cached)
                with torch.no_grad():
                    if not args.not_cache_latents:
                        # Latents & text from cached dataset - already sampled
                        latents = batch[0]  # This is now the sampled latent, not the distribution
                        if args.train_text_encoder:
                            input_ids_or_hidden_states = batch[1]  # input_ids
                        else:
                            input_ids_or_hidden_states = batch[1]  # hidden_states
                    else:
                        # Standard route
                        batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)
                        batch["input_ids"] = batch["input_ids"].to(accelerator.device)
                        latent_dist = vae.encode(batch["pixel_values"]).latent_dist
                        latents = latent_dist.sample() * 0.18215
                        input_ids_or_hidden_states = batch["input_ids"]

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Text encoder
                with text_enc_context:
                    if not args.not_cache_latents:
                        if args.train_text_encoder:
                            encoder_hidden_states = text_encoder(input_ids_or_hidden_states)[0]
                        else:
                            encoder_hidden_states = input_ids_or_hidden_states
                    else:
                        encoder_hidden_states = text_encoder(input_ids_or_hidden_states)[0]

                # Predict noise
                model_pred = transformer(
                    sample=noisy_latents,
                    time_ids=timesteps,  # in SD3.5, "time_ids" is the correct parameter
                    encoder_hidden_states=encoder_hidden_states
                ).sample

                # Depending on the scheduler's prediction_type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # If using prior preservation, half of the batch is instance, half is class
                if args.with_prior_preservation and args.class_data_dir and args.class_prompt:
                    # Split model_pred and target into two halves
                    half = bsz // 2
                    model_pred, model_pred_prior = model_pred[:half], model_pred[half:]
                    target, target_prior = target[:half], target[half:]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.is_main_process:
                if step % 10 == 0:
                    progress_bar.set_postfix({"loss": loss.item()})
            progress_bar.update(1)
            global_step += 1

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()
        if global_step >= args.max_train_steps:
            break

    # -------------------------------------------------------------------------
    # 5. Final save
    # -------------------------------------------------------------------------
    if accelerator.is_main_process:
        logger.info("** Saving final pipeline weights **")

        # If text encoder was trained, unwrap it
        if args.train_text_encoder:
            final_text_encoder = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
        else:
            final_text_encoder = CLIPTextModel.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="text_encoder",
                revision=args.revision
            )

        final_transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=True)

        # Construct the final pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=final_transformer,
            text_encoder=final_text_encoder,
            vae=AutoencoderKL.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="vae",
                revision=args.revision,
                use_auth_token=huggingface_token,
            ),
            safety_checker=None,
            torch_dtype=torch.float16,
            revision=args.revision,
            use_auth_token=huggingface_token,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        if is_xformers_available():
            pipe.enable_xformers_memory_efficient_attention()

        pipe.save_pretrained(args.output_dir)
        logger.info(f"Model saved at {args.output_dir}")

        # Optionally, generate sample images
        if args.save_sample_prompt:
            logger.info("Generating sample images...")
            pipe.to(accelerator.device)
            g_cuda = torch.Generator(device=accelerator.device)
            if args.seed is not None:
                g_cuda.manual_seed(args.seed)

            sample_dir = os.path.join(args.output_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)

            with torch.autocast("cuda"), torch.inference_mode():
                for i in tqdm(range(args.n_save_sample), desc="Sample generation"):
                    images = pipe(
                        args.save_sample_prompt,
                        negative_prompt=args.save_sample_negative_prompt,
                        guidance_scale=args.save_guidance_scale,
                        num_inference_steps=args.save_infer_steps,
                        generator=g_cuda
                    ).images
                    images[0].save(os.path.join(sample_dir, f"sample_{i}.png"))

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)