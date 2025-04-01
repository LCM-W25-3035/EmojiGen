from fastapi import FastAPI, HTTPException
import torch
import io
import base64
import uvicorn
from models import Generator
from schemas import GenerateRequest, GenerateResponse
from utilities import embed_text, tensor_to_pil, device

current_dir = os.path.dirname(os.path.abspath(__file__))
emoji_model_path = os.path.join(current_dir, "models", "cgan_emoji_generator.pth")
sticker_model_path = os.path.join(current_dir, "models", "cgan_sticker_generator.pth")

# Instantiate models for emoji and sticker generation
emoji_generator = Generator().to(device)
sticker_generator = Generator().to(device)

# Loading pre-trained models
emoji_generator.load_state_dict(torch.load(emoji_model_path, map_location=device))
sticker_generator.load_state_dict(torch.load(sticker_model_path, map_location=device))
emoji_generator.eval()
sticker_generator.eval()

# Create FastAPI app instance
app = FastAPI()

@app.post("/generate", response_model=GenerateResponse)
def generate_image(req: GenerateRequest):
    if req.model_type.lower() not in ["emoji", "sticker"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'emoji' or 'sticker'.")
    
    # Generate text embedding using CLIP
    embed = embed_text(req.prompt)  # shape: (1, 512)
    # Generate random noise vector
    noise = torch.randn(1, 100).to(device)
    
    # Generate image using the selected generator model
    with torch.no_grad():
        if req.model_type.lower() == "emoji":
            gen_tensor = emoji_generator(noise, embed)
        else:
            gen_tensor = sticker_generator(noise, embed)
    
    # Convert tensor output to a PIL image
    pil_image = tensor_to_pil(gen_tensor)
    
    # Convert PIL image to a base64-encoded PNG
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return GenerateResponse(image_base64=img_str)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
