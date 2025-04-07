import torch
from transformers import CLIPTokenizer, CLIPTextModel
import numpy as np
import io
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CLIP components for text embedding
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

def embed_text(prompt: str):
    """
    Embed the text prompt using CLIP.
    Returns a tensor of shape (1, 512).
    """
    inputs = clip_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=77).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
    input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled  # shape: (1, 512)

def tensor_to_pil(image_tensor):
    """
    Convert an image tensor to a PIL image.
    Assumes image_tensor shape is (1, 3, H, W) with range [-1, 1].
    """
    image_tensor = (image_tensor + 1) / 2  # scale to [0, 1]
    image_tensor = image_tensor.clamp(0, 1)
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)
