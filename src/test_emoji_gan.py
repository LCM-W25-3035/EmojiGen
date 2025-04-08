import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.models import Inception_V3_Weights
from scipy.linalg import sqrtm
import sys
import argparse
from transformers import CLIPTokenizer, CLIPTextModel

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate emojis from text prompts.")
parser.add_argument("--prompts", nargs="+", default=[
    "Crying face",
    "A brown man running in the left direction",
    "An angry red face",
    "A face wearing sunglasses depicting a sense of coolness",
    "A man and a woman in love"
], help="List of text prompts to generate emojis.")
parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the generated emoji plot.")
parser.add_argument("--model_path", type=str, default="../saved_models/cgan_model_generator.pth", help="Path to the saved generator model.")
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Load CLIP's tokenizer and text model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model = clip_model.to(device)
clip_model.eval()

def mean_pooling(model_output, attention_mask):
    """Mean pool the token embeddings."""
    token_embeddings = model_output.last_hidden_state  # (batch_size, sequence_length, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

def embed_text(text):
    """Embed the input text using CLIP."""
    # Tokenize the input text
    inputs = clip_tokenizer(text, return_tensors="pt", truncation=True, max_length=77)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Disable gradients for inference
    with torch.no_grad():
        output = clip_model(**inputs)
    
    # Pool the token embeddings (mean pooling)
    pooled_embedding = mean_pooling(output, inputs["attention_mask"])
    
    # L2 normalize the pooled embedding
    pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)
    
    return pooled_embedding.squeeze().cpu().numpy().astype(np.float32)


#Generator model
class Generator(torch.nn.Module):
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

# Initialize the generator model
noise_dim = 100  # Adjust based on your generator's input noise dimension
embedding_dim = 512  # CLIP text embedding dimension
generator = Generator(noise_dim, embedding_dim).to(device)

# Load the pre-trained generator weights
generator.load_state_dict(torch.load("../saved_models/cgan_emoji_generator.pth", map_location=device))
generator.eval()

# Generate emojis for the provided prompts
prompts = args.prompts
generated_images = []

for prompt in prompts:
    # Embed the text prompt
    text_embedding = embed_text(prompt)
    text_embedding = torch.tensor(text_embedding).to(device).unsqueeze(0)
    
    # Generate a random noise vector
    noise = torch.randn(1, noise_dim).to(device)
    
    # Generate the emoji image
    with torch.no_grad():
        gen_image = generator(noise, text_embedding)
        gen_image = gen_image.cpu()  # move to CPU for plotting
    generated_images.append(gen_image)

# Function to convert a tensor image to a numpy image (assumes [1, C, H, W] in range [-1, 1])
def tensor_to_image(tensor):
    image = tensor.squeeze(0)          # remove batch dimension
    image = (image + 1) / 2            # scale to [0, 1]
    image = image.permute(1, 2, 0).numpy()  # convert to HWC format
    return np.clip(image, 0, 1)

# Plot the generated emojis along with their corresponding prompts
fig, axs = plt.subplots(1, len(prompts), figsize=(20, 4))
for i, (prompt, gen_img) in enumerate(zip(prompts, generated_images)):
    img_np = tensor_to_image(gen_img)
    axs[i].imshow(img_np)
    axs[i].set_title(prompt, fontsize=10)
    axs[i].axis("off")
plt.suptitle("Emojis Generated from Text Prompts", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the figure to disk
save_path = os.path.join(args.output_dir, "emojis_from_prompts.png")
plt.savefig(save_path)
print(f"Saved generated emojis plot at: {save_path}")
plt.close()
