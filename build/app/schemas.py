from pydantic import BaseModel

class GenerateRequest(BaseModel):
    prompt: str
    model_type: str  # "emoji" or "sticker"

class GenerateResponse(BaseModel):
    image_base64: str
