import os
import torch
import numpy as np
from torch import nn
from pathlib import Path
from transformers import CLIPTokenizer, CLIPTextModel

class Generator(nn.Module):
    def __init__(self, noise_dim, embedding_dim, image_channels=3):
        super(Generator, self).__init__()
        self.noise_fc = nn.Sequential(
            nn.Linear(noise_dim, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.embed_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256 * 4 * 4),
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
        embed_features = self.embed_fc(embed).view(embed.size(0), 256, 4, 4)
        x = noise_features + embed_features
        x = self.conv_blocks(x)
        return x

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize generator
    noise_dim = 100
    embedding_dim = 512  # CLIP embedding dimension
    generator = Generator(noise_dim, embedding_dim).to(device)
    
    # Load weights
    model_path = os.path.join("models", "cgan_emoji_generator.pth")
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()
    
    # Load CLIP model
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    
    return generator, clip_model, clip_tokenizer, device

def generate_emoji(text, generator, clip_model, clip_tokenizer, device, noise_dim=100):
    """Generate emoji from text prompt"""
    # Get CLIP embedding
    inputs = clip_tokenizer(text, return_tensors="pt", truncation=True, max_length=77)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        output = clip_model(**inputs)
        pooled_embedding = output.last_hidden_state.mean(dim=1)  # Simple mean pooling
        pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)
    
    # Generate noise
    noise = torch.randn(1, noise_dim).to(device)
    
    # Generate image
    with torch.no_grad():
        gen_image = generator(noise, pooled_embedding)
    
    return gen_image