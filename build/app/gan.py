import io
import os
import torch
import base64
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.init as init
from transformers import CLIPTokenizer, CLIPTextModel

class EmojiGenerator(nn.Module):
    def __init__(self, noise_dim, embedding_dim, image_channels=3):
        super(EmojiGenerator, self).__init__()
        # Map noise vector to feature map
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 256 * 4 * 4),
            nn.ReLU()
        )
        # Map text embedding to feature map
        self.embed_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256 * 4 * 4),
            nn.ReLU()
        )
        # Upsampling blocks to generate an image
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, embed):
        noise_features = self.noise_fc(noise).view(noise.size(0), 256, 4, 4)
        embed_features = self.embed_fc(embed).view(embed.size(0), 256, 4, 4)
        x = noise_features + embed_features
        return self.conv_blocks(x)
    
class StickerGenerator(nn.Module):
    def __init__(self, noise_dim, embedding_dim=512, image_channels=3):
        super(StickerGenerator, self).__init__()
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 256 * 4 * 4),
            nn.ReLU(),
        )
        
        # embed_transform layer should accept embedding_dim=512 from CLIP
        self.embed_transform = nn.Linear(embedding_dim, 384)  # embedding_dim = 512 to match CLIP output

        self.embed_fc = nn.Sequential(
            nn.Linear(384, 256 * 4 * 4),  # Adjusted to match embed_transform output size
            nn.ReLU(),
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, embed):
        noise_features = self.noise_fc(noise).view(noise.size(0), 256, 4, 4)
        embed = self.embed_transform(embed)  # Apply transformation
        embed_features = self.embed_fc(embed).view(embed.size(0), 256, 4, 4)
        x = noise_features + embed_features
        x = self.conv_blocks(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_dir = os.path.dirname(os.path.abspath(__file__))
emoji_model_path = os.path.join(current_dir, "gan_models", "cgan_emoji_generator.pth")
sticker_model_path = os.path.join(current_dir, "gan_models", "cgan_sticker_generator.pth")

# Settings for CGAN
noise_dim = 100
emoji_embedding_dim = 512
sticker_embedding_dim = 384

# Loading pre-trained CGAN models
emoji_generator = EmojiGenerator(noise_dim, emoji_embedding_dim).to(device)
emoji_generator.load_state_dict(torch.load(emoji_model_path, map_location=device))
emoji_generator.eval()

# sticker_generator = StickerGenerator(noise_dim, sticker_embedding_dim).to(device)
# sticker_generator.load_state_dict(torch.load(sticker_model_path, map_location=device))
# sticker_generator.eval()

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

def gan(prompt, image_type):
    # Generate text embedding using CLIP
    embed = embed_text(prompt)  # shape: (1, 512)
    
    # Generate random noise vector
    noise = torch.randn(1, 100).to(device)
    
    with torch.no_grad():
        if image_type.lower() == "emoji":
            gen_tensor = emoji_generator(noise, embed)
        else:
            # gen_tensor = sticker_generator(noise, embed)
            pass
        
    # Convert tensor output to a PIL image
    pil_image = tensor_to_pil(gen_tensor)
        
    # Convert PIL image to a base64-encoded PNG
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    return img_base64