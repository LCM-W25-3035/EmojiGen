import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np

# Enable cuDNN optimization for faster training
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Ensure PyTorch uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Free GPU memory
torch.cuda.empty_cache()
gc.collect()


# Dataset Class
class EmojiDataset(Dataset):
    def __init__(self, parquet_file):
        self.data = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["image_path"]
        image_tensor = torch.load(image_path).float()  # No manual normalization
        text_embedding = torch.tensor(self.data.iloc[idx]["combined_embedding"], dtype=torch.float32)
        return image_tensor, text_embedding

# Load Dataset
parquet_file = "../data/processed_emoji_dataset.parquet"
dataset = EmojiDataset(parquet_file)
batch_size = 32  
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Load Stable Diffusion 2 Base Model
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="vae").to(device, dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="unet").to(device, dtype=torch.float16)

# Apply LoRA to UNet
lora_config = LoraConfig(
    r=1,  # LoRA rank
    lora_alpha=8,  # Scaling factor
    target_modules=["to_q", "to_k", "to_v", "proj_out", "proj_in"],  
    lora_dropout=0.1,  # Dropout for regularization
    bias="none"
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()

# Enable memory optimization
unet.enable_gradient_checkpointing()

class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024):  # Change output to 1280
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

embedding_projector = EmbeddingProjector().to(device, dtype=torch.float16)

# Freeze VAE
for param in vae.parameters():
    param.requires_grad = False  

# Define optimizer
optimizer = AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=0.001)
scaler = torch.amp.GradScaler()


# Training Loop
num_epochs = 30
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, embeddings in progress_bar:
        images = images.to(device, dtype=torch.float16, non_blocking=True)
        embeddings = embeddings.to(device, dtype=torch.float16, non_blocking=True)
        optimizer.zero_grad()

        # Project CLIP embeddings
        with torch.no_grad():
            projected_embeddings = embedding_projector(embeddings).unsqueeze(1)
        
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215
        latents = latents.to(torch.float16)

        # Generate noise
        noise = torch.randn_like(latents, dtype=torch.float16)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()

        # Forward pass
        with torch.amp.autocast("cuda"):
            noise_pred = unet(latents, timesteps, encoder_hidden_states=projected_embeddings).sample
            loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

# Save Model
torch.save({
    "unet": unet.state_dict(),
    "embedding_projector": embedding_projector.state_dict()
}, "emoji_generator.pth")
print("Model saved successfully!")


# Save LoRA Weights
unet.save_pretrained("lora_emoji_unet")
torch.save(embedding_projector.state_dict(), "embedding_projector.pth")
print("LoRA adapters saved successfully!")

# Plot Loss Curve
plt.plot(range(1, num_epochs + 1), losses, marker="o", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()

# Load Model for Inference
unet = PeftModel.from_pretrained(unet, "lora_emoji_unet").to(device).to(torch.float16)
embedding_projector.load_state_dict(torch.load("embedding_projector.pth"))
unet.eval()
embedding_projector.eval()



# Load CLIP Model
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Text Processing
text_description = "Smiling face"
tokens = tokenizer(text_description, return_tensors="pt").to(device)
text_embedding = text_encoder(**tokens).last_hidden_state[:, 0, :]  # Use CLS token
text_embedding = text_embedding.to(torch.float16)
projected_embedding = embedding_projector(text_embedding).unsqueeze(0).to(torch.float16)

# Generate Noise
latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
timesteps = torch.tensor([500], device=device).long()



# Generate Emoji
with torch.no_grad():
    denoised_latents = unet(latents, timesteps, encoder_hidden_states=projected_embedding).sample

denoised_latents = denoised_latents / 0.18215

with torch.no_grad():
    decoded_image = vae.decode(denoised_latents).sample

decoded_image = (decoded_image.clamp(-1, 1) + 1) / 2
emoji_image = Image.fromarray((decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
emoji_image.show()



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, PeftModel


# ### Prompt(GPT 4o): Fine tunning Stable Difussion model using Lora .


# Enable cuDNN optimization for faster training
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Ensure PyTorch uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Free GPU memory
torch.cuda.empty_cache()
gc.collect()



# Dataset Class
class EmojiDataset(Dataset):
    def __init__(self, parquet_file):
        self.data = pd.read_parquet(parquet_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]["image_path"]
        image_tensor = torch.load(image_path).float() / 127.5 - 1  # Normalize to [-1,1]
        text_embedding = torch.tensor(self.data.iloc[idx]["combined_embedding"], dtype=torch.float32)
        return image_tensor, text_embedding


# Load Dataset
parquet_file = "../data/processed_emoji_dataset.parquet"
dataset = EmojiDataset(parquet_file)
batch_size = 32  
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load Stable Diffusion 2 Base Model (Trained on 256x256 images)
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="vae").to(device, dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="unet").to(device, dtype=torch.float16)

print(unet.config) 

# Apply LoRA to UNet
lora_config = LoraConfig(
    r=1,  # LoRA rank
    lora_alpha=8,  # Scaling factor
    target_modules=["to_q", "to_k", "to_v", "proj_out", "proj_in"],  
    lora_dropout=0.1,  # Dropout for regularization
    bias="none"
)

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()  # Print trainable parameters (should be very small)



# Enable memory optimization
unet.enable_gradient_checkpointing()

# Embedding Projector (CLIP 512 → UNet 768)
class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024):  # Changed to 768
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

embedding_projector = EmbeddingProjector().to(device, dtype=torch.float16)

# Freeze VAE
for param in vae.parameters():
    param.requires_grad = False  

# Define optimizer
optimizer = AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=0.001)
scaler = torch.amp.GradScaler()



# Training Loop
num_epochs = 5
losses = []
for epoch in range(num_epochs):
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, embeddings in progress_bar:
        images = images.to(device, dtype=torch.float16, non_blocking=True)
        embeddings = embeddings.to(device, dtype=torch.float16, non_blocking=True)
        optimizer.zero_grad()

        # Project CLIP embeddings
        with torch.no_grad():
            projected_embeddings = embedding_projector(embeddings).unsqueeze(1)
        
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * 0.18215

        # Convert latents to bfloat16 to save memory
        latents = latents.to(torch.float16)

        # Generate noise
        noise = torch.randn_like(latents, dtype=torch.float16)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()

        # Forward pass
        with torch.amp.autocast("cuda"):
            noise_pred = unet(latents, timesteps, encoder_hidden_states=projected_embeddings).sample
            loss = F.mse_loss(noise_pred, noise)

        # Backpropagation
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # Avoid NaN issues
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")


# Save Model
torch.save({
    "unet": unet.state_dict(),
    "embedding_projector": embedding_projector.state_dict()
}, "emoji_generator.pth")
print("Model saved successfully!")

# Save LoRA Weights
unet.save_pretrained("lora_emoji_unet")
torch.save(embedding_projector.state_dict(), "embedding_projector.pth")
print("LoRA adapters saved successfully!")


# Plot Loss Curve
plt.plot(range(1, num_epochs + 1), losses, marker="o", linestyle="-")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid()
plt.show()


unet = PeftModel.from_pretrained(unet, "lora_emoji_unet").to(device).to(torch.float16)
embedding_projector.load_state_dict(torch.load("embedding_projector.pth"))
unet.eval()
embedding_projector.eval()


# Assuming the necessary models are already loaded
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = text_encoder.to(device)


# Define your text prompt
text_description = "Smiling face"  # This is where we specify "dog"

# Tokenize and encode the text prompt
tokens = tokenizer(text_description, return_tensors="pt").to(device)
text_embedding = text_encoder(**tokens).last_hidden_state.mean(dim=1)  # Aggregate token embeddings
text_embedding = text_embedding.to(torch.float16)  # Ensure it's float16 for compatibility with UNet

# Project the embedding to match UNet’s expected format
projected_embedding = embedding_projector(text_embedding).unsqueeze(0).to(torch.float16)  # Ensure float16

# Generate noise in latent space (fixed size 96x96)
latents = torch.randn(1, 4, 96, 96).to(device).to(torch.float16)  # Ensure latents are in float16
timesteps = torch.tensor([500], device=device).long()
torch.cuda.empty_cache()
gc.collect()

# Generate noise in latent space (smaller size)
latents = torch.randn(1, 4, 64, 64, device=device, dtype=torch.float16)
timesteps = torch.tensor([500], device=device).long()


# Generate Emoji
with torch.no_grad():
    denoised_latents = unet(latents, timesteps, encoder_hidden_states=projected_embedding).sample

# Move to CPU and Decode
denoised_latents = denoised_latents / 0.18215

with torch.no_grad():
    decoded_image = vae.decode(denoised_latents).sample


# Post-process Image
decoded_image = (decoded_image.clamp(-1, 1) + 1) / 2
decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
decoded_image = (decoded_image * 255).astype(np.uint8)
emoji_image = Image.fromarray(decoded_image)

# Display the image
emoji_image.show()





