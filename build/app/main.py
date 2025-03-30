from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
import base64
import numpy as np
import torch
from pydantic import BaseModel
from typing import List
import os

from app.models import load_model, generate_emoji
from app.schemas import GenerationRequest

app = FastAPI(title="Emoji Generator API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
@app.on_event("startup")
async def startup_event():
    global generator, clip_model, clip_tokenizer, device
    generator, clip_model, clip_tokenizer, device = load_model()

class TextRequest(BaseModel):
    text: str

@app.post("/generate-from-text")
async def generate_from_text(request: TextRequest):
    try:
        # Generate emoji from text
        image_tensor = generate_emoji(
            request.text, 
            generator, 
            clip_model, 
            clip_tokenizer, 
            device
        )
        
        # Convert tensor to PIL Image
        image = tensor_to_image(image_tensor)
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return {"image": img_str}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def tensor_to_image(tensor):
    """Convert a tensor image to PIL Image"""
    image = tensor.squeeze(0).cpu()  # remove batch dim and move to cpu
    image = (image + 1) / 2  # scale from [-1, 1] to [0, 1]
    image = image.permute(1, 2, 0).numpy()  # CHW to HWC
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)