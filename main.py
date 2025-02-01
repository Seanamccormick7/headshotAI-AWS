# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from tasks import generate_images_task

app = FastAPI()

class GenerateRequest(BaseModel):
    userId: str
    gender: str = None
    hairColor: str = None
    hairLength: str = None
    ethnicity: str = None
    bodyType: str = None
    attire: str = None
    backgrounds: str = None
    glasses: bool = False
    instanceImages: List[str] = []
    callbackUrl: str

@app.post("/generate")
def generate_images(req: GenerateRequest):
    try:
        # Enqueue the long-running generation task.
        task = generate_images_task.delay(req.dict())
        return {"message": "Generation request enqueued.", "task_id": task.id}
    except Exception as e:
        print("Error in /generate endpoint:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
