import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import Request, Response
from app.diffusion import emoji_diffusion
from app.gan import gan

# Create FastAPI app instance
app = FastAPI()

# CORS configuration
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
 )

@app.post("/generate", response_model=Response)
def generate_image(req: Request):
    if req.gen_model.lower() not in ["diffusion", "gan"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'gan' or 'diffusion'.")
    
    if req.image_type.lower() not in ["emoji", "sticker"]:
        raise HTTPException(status_code=400, detail="Invalid image type. Choose 'emoji' or 'sticker'.")
    
    if req.gen_model.lower() == "diffusion":
        try:
            img_b64 = emoji_diffusion(req.prompt, req.image_type)
            return Response(image=img_b64)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        try:
            img_b64 = gan(req.prompt, req.image_type)
            return Response(image=img_b64)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
