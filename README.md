# Headshot AI Backend

> Production-ready AI headshot generation service built with FastAPI, Celery, and Stable Diffusion 1.5

A high-performance backend service that trains custom DreamBooth models on Stable Diffusion 1.5 to generate professional-quality headshots from user photos. Optimized for GPU servers with enterprise-grade scalability.

## Overview

This service accepts multiple user photos and delivers professional headshots through an automated ML pipeline:

- **Custom Model Training**: DreamBooth fine-tuning on SD 1.5 with prior preservation
- **Distributed Processing**: Asynchronous task queue with Celery for concurrent job handling
- **Production-Ready**: Docker containerized with NVIDIA runtime support
- **Cloud-Optimized**: Designed for GPU instances with 16GB+ VRAM

## Key Features

- **Fast Training**: Configurable training steps (default: 800) on user-provided images
- **High Quality Output**: 512px professional headshots with customizable attributes
- **Scalable Architecture**: Stateless workers with Redis-backed task queue
- **Uploadcare Integration**: Direct CDN upload with automatic callback delivery
- **Configurable Pipeline**: Dynamic prompt generation based on user characteristics
- **Prior Preservation**: Maintains class consistency with 50 reference images

## Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   FastAPI   │────────▶│    Celery    │────────▶│  SD 1.5 +   │
│   (REST)    │         │   Workers    │         │  DreamBooth │
└─────────────┘         └──────────────┘         └─────────────┘
      │                        │                         │
      ▼                        ▼                         ▼
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│    Redis    │         │  Uploadcare  │         │  Generated  │
│  (Broker)   │         │    (CDN)     │         │   Images    │
└─────────────┘         └──────────────┘         └─────────────┘
```

## Project Structure

```
.
├── class_images/                  # Prior preservation dataset (~50 images)
├── dreambooth/
│   ├── concepts_list.json         # Training configuration
│   ├── requirements.txt           # Python dependencies
│   └── train_dreambooth.py        # DreamBooth training script
├── .dockerignore
├── .gitignore
├── Dockerfile                     # Multi-stage build with CUDA 11.8
├── README.md
├── celery_app.py                  # Celery configuration and initialization
├── main.py                        # FastAPI application and routes
├── predict.py                     # Training orchestration wrapper
└── tasks.py                       # Distributed task implementation
```

## Quick Start

### Prerequisites

- NVIDIA GPU with 16GB+ VRAM (tested on A10G, T4, V100)
- NVIDIA Driver 470+ (CUDA 11.8 compatible)
- Docker with NVIDIA Container Toolkit
- Redis server (local or cloud-hosted)

### Environment Configuration

Create a `.env` file:

```bash
# Uploadcare Credentials
UPLOADCARE_PUBLIC_KEY=your_public_key
UPLOADCARE_SECRET_KEY=your_secret_key

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Model Cache (optional)
HF_HOME=/root/.cache/huggingface
```

### Build and Deploy

```bash
# Build the Docker image
docker build -t headshotai-backend .

# Start FastAPI server
docker run --rm -it --gpus all --env-file ./.env \
  -p 8080:8080 \
  --name headshot-api \
  headshotai-backend \
  uvicorn main:app --host 0.0.0.0 --port 8080

# Start Celery worker (separate terminal)
docker run --rm -it --gpus all --env-file ./.env \
  --name headshot-worker \
  headshotai-backend \
  celery -A celery_app.celery_app worker --loglevel=INFO
```

## API Reference

### POST `/generate`

Initiates the training and inference pipeline.

**Request Body:**

```json
{
  "userId": "user_123",
  "instanceImages": [
    "uploadcare-uuid-1",
    "uploadcare-uuid-2",
    "uploadcare-uuid-3",
    "uploadcare-uuid-4",
    "uploadcare-uuid-5"
  ],
  "callbackUrl": "https://your-frontend.com/api/callback",
  "gender": "man",
  "age": 30,
  "hairColor": "brown",
  "hairLength": "short",
  "ethnicity": "latino",
  "bodyType": "average",
  "attire": "business suit",
  "backgrounds": "studio backdrop",
  "glasses": false
}
```

**Response:** `200 OK` with task acknowledgment

```json
{
  "message": "Generation request enqueued.",
  "task_id": "abc123-task-id"
}
```

**Callback Payload:**

The service will POST to your `callbackUrl` with:

```json
{
  "userId": "user_123",
  "generatedUuids": [
    "188f5bb5-8474-4e73-a549-2d04e875af45",
    "65e23a95-4242-4547-bea3-a678dc32e0d9"
  ]
}
```

## Training Configuration

Optimized hyperparameters for facial likeness and quality:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base Model | `runwayml/stable-diffusion-v1-5` | SD 1.5 |
| VAE | `stabilityai/sd-vae-ft-mse` | Fine-tuned VAE |
| Resolution | 512px | Native SD 1.5 resolution |
| Learning Rate | 1e-6 | Conservative for stability |
| Training Steps | 800 | Configurable via API |
| Batch Size | 1 | VRAM optimized |
| Gradient Accumulation | 1 | Increase for memory constraints |
| Mixed Precision | fp16 | 2x memory efficiency |
| Text Encoder Training | Enabled | Improved prompt adherence |
| Prior Preservation | Enabled | Class consistency |
| 8-bit Adam | Enabled | Memory optimization |

### Dynamic Prompt Generation

The system automatically generates training prompts based on user characteristics:

```python
instance_prompt = (
    f"Photo of a {age}-year-old {gender} with {hairColor} hair "
    f"({hairLength}), of {ethnicity} ethnicity, {bodyType} build, {glasses_status}."
)

sample_prompt = (
    f"High quality professional headshot of a {age}-year-old {gender} with {hairColor} hair "
    f"({hairLength}), of {ethnicity} ethnicity, {bodyType} build, {glasses_status}, "
    f"wearing {attire}, in a {backgrounds} setting."
)
```

## Pipeline Workflow

1. **Download**: Fetch instance images from Uploadcare CDN as JPEGs
2. **Preparation**: Zip images and prepare training directory
3. **Training**: Launch DreamBooth fine-tuning with:
   - Prior preservation using 50 reference images
   - Text encoder training for better prompt alignment
   - Mixed precision fp16 training with 8-bit Adam
4. **Generation**: Model automatically generates sample images during training
5. **Upload**: Push generated images to Uploadcare CDN
6. **Callback**: POST image UUIDs to client webhook
7. **Cleanup**: Remove temporary training data and free resources

## Performance Optimization

### Speed Improvements

- **Model Caching**: First run downloads models (~4GB), subsequent runs use cached weights
- **Reduced Steps**: Lower training steps to 400-600 for 2x speed (slight quality tradeoff)
- **Persistent Storage**: Mount `HF_HOME` for cross-container cache sharing
- **8-bit Optimization**: Uses 8-bit Adam optimizer for reduced memory footprint

### Quality Tuning

- **Training Steps**: Increase from 800 → 1000-1200 for better likeness
- **Sample Quality**: Adjust `save_guidance_scale` (default: 7.5) and `save_infer_steps` (default: 20)
- **Prompt Engineering**: System automatically incorporates detailed user characteristics

### Memory Management

Current configuration fits comfortably on 16GB VRAM. For lower memory cards:

- Increase `gradient_accumulation_steps` to 2-4
- Keep `use_8bit_adam` enabled (already enabled)
- Reduce `train_batch_size` (already at 1)

## Customization

### Adjusting Training Parameters

Edit `predict.py`:

```python
cmd = [
    "python", "dreambooth/train_dreambooth.py",
    "--learning_rate=1e-6",        # Lower = stable, higher = faster
    f"--max_train_steps={steps}",  # More steps = better likeness
    "--gradient_accumulation_steps=1",  # Increase for memory
    "--save_interval=400",          # Checkpoint frequency
]
```

### Modifying Generation Settings

Edit `predict.py` sample generation parameters:

```python
run_training(
    steps=800,                      # Total training steps
    save_guidance_scale=7.5,        # Prompt adherence (5-15 range)
    save_infer_steps=20,            # Generation quality
    n_save_sample=2                 # Images per checkpoint
)
```

### Custom Prompt Templates

Edit `tasks.py` to modify the prompt construction logic for different styles, poses, or artistic directions.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA Out of Memory** | Increase `gradient_accumulation_steps` or enable gradient checkpointing |
| **Slow First Run** | Model download takes ~10min; cache with mounted `HF_HOME` |
| **No Images Generated** | Check `samples_dir` path matches training script output |
| **Callback Failed** | Verify `callbackUrl` is accessible and accepts POST requests |
| **Poor Likeness** | Increase training steps from 800 → 1000+ |
| **Redis Connection Error** | Ensure Redis is running and `REDIS_URL` is correct |

## Technical Specifications

- **Framework**: FastAPI 0.95+, Celery 5.2+
- **ML Stack**: PyTorch 2.0.1 (CUDA 11.8), Diffusers 0.28.2, Accelerate 0.30.1
- **Base Model**: Stable Diffusion 1.5 (860M parameters)
- **Training Method**: DreamBooth with prior preservation
- **Memory Footprint**: ~14GB VRAM during training, ~8GB during inference
- **Throughput**: ~10-15 minutes per complete job (800 steps + image generation)

## Security & Compliance

- **Consent Required**: Verify explicit user consent before generating images
- **Content Moderation**: Implement upstream filters for inappropriate requests
- **Privacy**: All training data is ephemeral and deleted post-generation
- **License Compliance**: SD 1.5 follows CreativeML OpenRAIL-M license
- **Rate Limiting**: Implement API throttling to prevent abuse

## Dependencies

Core libraries (from `dreambooth/requirements.txt`):

```
accelerate==0.30.1
diffusers==0.28.2
transformers==4.38.2
torch==2.0.1+cu118
torchvision==0.15.2+cu118
xformers==0.0.22.post7
fastapi>=0.95.0
uvicorn>=0.21.1
celery>=5.2.7
redis>=4.3.4
pyuploadcare>=6.0.0
bitsandbytes>=0.41.1
```

See `dreambooth/requirements.txt` for complete dependency list.

## Docker Configuration

The Dockerfile uses a multi-stage approach:

- **Base**: Python 3.10-slim
- **System Dependencies**: git, libgl1-mesa-glx, libglib2.0-0, unzip
- **PyTorch**: CUDA 11.8 build from official PyTorch repository
- **Port**: 8080 for FastAPI application
- **Optimizer**: hf_transfer for faster Hugging Face downloads

## License

This project is a custom implementation built on:

- **Diffusers** - Apache 2.0 License
- **Stable Diffusion 1.5** - CreativeML OpenRAIL-M License
- **DreamBooth** - Research paper implementation

Review individual component licenses before commercial deployment.

## Acknowledgments

Built with Hugging Face Diffusers DreamBooth implementation and Stability AI's Stable Diffusion 1.5 model.

---

**Note**: This is a production-grade implementation designed for enterprise deployments. Ensure proper resource allocation, monitoring, and compliance with applicable ML model licenses.
