
# HeadshotAI Backend (AWS)

A **production-ready backend** for an AI-powered headshot generator.  
This service accepts user uploads and preferences, runs model training/inference, and returns professional-quality headshots. Built with modern backend tooling and deployed on AWS.

---

## 🚀 Features

- **FastAPI API layer** for handling requests (`/generate`, job status, history).
- **Task orchestration with Celery** for long-running jobs (non-blocking API responses).
- **Background workers** for model training + inference (DreamBooth on Stable Diffusion 1.5).
- **File management** via Uploadcare (upload, package, and return results).
- **Database integration** with PostgreSQL for users, orders, and job tracking.
- **Secure and production-ready**:
  - Dockerized backend and worker services
  - Configurable with environment variables
  - Ready for scaling on AWS EC2 or ECS

---

## Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Task Queue**: [Celery](https://docs.celeryq.dev/)
- **Model Training**: DreamBooth fine-tuning on **Stable Diffusion 1.5**
- **File Handling**: [Uploadcare](https://uploadcare.com/)
- **Database**: PostgreSQL
- **Infrastructure**: AWS EC2, Docker
- **Language**: Python 3.10+

---

## Installation & Setup

Clone the repo:

```bash
git clone git@github.com:Seanamccormick7/headshotAI-AWS.git
cd headshotAI-AWS
```

Create your .env file
# .env example
DATABASE_URL=postgresql://user:password@host:5432/dbname
BROKER_URL=redis://localhost:6379/0
UPLOADCARE_PUBLIC_KEY=your_public_key
UPLOADCARE_SECRET_KEY=your_secret_key

Build and run with Docker:
docker compose up --build

## System Architecture
flowchart TD
    A[Frontend Uploads & Preferences] -->|POST /generate| B[FastAPI Backend]
    B -->|Task Enqueue| C[Celery Worker]
    C -->|Download Files| D[Uploadcare]
    C -->|Fine-Tune & Inference| E[DreamBooth / Stable Diffusion]
    E -->|Upload Results| D
    C -->|Callback Update| B
    B -->|Persist| F[PostgreSQL]

## Authors 
Created by Sean McCormick and Rayden Khuraijam.

## License

This project is licensed under the MIT License.