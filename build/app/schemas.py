from pydantic import BaseModel

class GenerationRequest(BaseModel):
    text: str