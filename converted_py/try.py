import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np

# Enable cuDNN optimization
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# Ensure GPU is used if available
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

# Train-Validation Split
train_size = int(0.9 * len(dataset))  # 90% train, 10% validation
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 32  
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



# Load Models
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="vae").to(device, dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="unet").to(device, dtype=torch.float16)

# Apply LoRA
lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["to_q", "to_k", "to_v", "proj_out", "proj_in"], lora_dropout=0.1, bias="none")
unet = get_peft_model(unet, lora_config)
unet.enable_gradient_checkpointing()

# Embedding Projector
class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim=512, output_dim=1024):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

embedding_projector = EmbeddingProjector().to(device, dtype=torch.float16)

# Freeze VAE
for param in vae.parameters():
    param.requires_grad = False  

# Define optimizer
optimizer = AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=1e-4)
scaler = torch.amp.GradScaler()



from tqdm import tqdm

# Training Loop
num_epochs = 30
best_val_loss = float('inf')
losses, val_losses = [], []
for epoch in range(num_epochs):
    unet.train()
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100, leave=True)
    
    for images, embeddings in train_dataloader:
        images, embeddings = images.to(device, dtype=torch.float16), embeddings.to(device, dtype=torch.float16)
        optimizer.zero_grad()
        projected_embeddings = embedding_projector(embeddings).unsqueeze(1)
        
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.mode() * 0.18215
        
        noise = torch.randn_like(latents, dtype=torch.float16)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
        
        with torch.amp.autocast("cuda"):
            noise_pred = unet(latents, timesteps, encoder_hidden_states=projected_embeddings).sample
            loss = F.mse_loss(noise_pred, noise)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        progress_bar.update(1)

    avg_epoch_loss = epoch_loss / len(train_dataloader)
    losses.append(avg_epoch_loss)
    
    # Validation Loop
    unet.eval()
    val_loss = 0
    with torch.no_grad():
        for images, embeddings in val_dataloader:
            images, embeddings = images.to(device, dtype=torch.float16), embeddings.to(device, dtype=torch.float16)
            projected_embeddings = embedding_projector(embeddings).unsqueeze(1)
            latents = vae.encode(images).latent_dist.mode() * 0.18215
            noise = torch.randn_like(latents, dtype=torch.float16)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()

            with torch.amp.autocast("cuda"):
                noise_pred = unet(latents, timesteps, encoder_hidden_states=projected_embeddings).sample
                loss = F.mse_loss(noise_pred, noise)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save Best Model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({"unet": unet.state_dict(), "embedding_projector": embedding_projector.state_dict()}, "best_emoji_generator.pth")
        print("Best model saved!")

# Plot Loss Curves
plt.plot(range(1, num_epochs + 1), losses, marker="o", linestyle="-", label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, marker="s", linestyle="--", label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss Curve")
plt.grid()
plt.show()

print("Training Complete!")
