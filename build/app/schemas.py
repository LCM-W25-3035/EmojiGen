from pydantic import BaseModel

class Request(BaseModel):
    prompt: str
    gen_model: str
    image_type: str

class Response(BaseModel):
    image: str
