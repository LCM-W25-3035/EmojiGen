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
import torch
from diffusers import UNet2DConditionModel
from peft import PeftModel

### Prompt(GPT 4o): Fine tunning Stable Difussion model using Lora .
#  Enable cuDNN optimization
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # Optimize matmul precision

#  Ensure PyTorch uses GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#  Free GPU memory
torch.cuda.empty_cache()
gc.collect()
#  Dataset Class
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
#  Load Dataset
parquet_file = "../data/processed_emoji_dataset.parquet"
dataset = EmojiDataset(parquet_file)
batch_size = 4  # Reduce batch size to free memory
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#  Load Stable Diffusion VAE and UNet to GPU
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae").to(device, dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet").to(device, dtype=torch.float16)
print(unet.config)  # Check image size
#  Apply LoRA to UNet
lora_config = LoraConfig(
    r=4,  # LoRA rank
    lora_alpha=16,  # Scaling factor
    target_modules=["to_q", "to_k", "to_v"],  # Apply LoRA to attention layers
    lora_dropout=0.05,  # Dropout for regularization
    bias="none"
    )

unet = get_peft_model(unet, lora_config)
unet.print_trainable_parameters()  #  Print trainable parameters (should be very small)
# Enable memory optimization
unet.enable_gradient_checkpointing()


# Embedding Projector (CLIP 512 → UNet 1024)
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

# Training Loop
num_epochs = 20
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
            latents = torch.utils.checkpoint.checkpoint(
                lambda x: vae.encode(x).latent_dist.sample() * 0.18215, images, use_reentrant=False
            )
        # Convert latents to bfloat16 to save memory
        latents = latents.to(torch.bfloat16)

        # Generate noise
        noise = torch.randn_like(latents, dtype=torch.bfloat16)
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

# Free GPU memory before inference
torch.cuda.empty_cache()
gc.collect()
# Load Trained LoRA Model
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet").to(device)
unet = PeftModel.from_pretrained(unet, "lora_emoji_unet").to(device).to(torch.float16)
embedding_projector.load_state_dict(torch.load("embedding_projector.pth"))
unet.eval()
embedding_projector.eval()
from transformers import CLIPTokenizer, CLIPTextModel
import torch
from PIL import Image
import numpy as np

# Assuming the necessary models are already loaded
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = text_encoder.to(device)

# Define your text prompt
text_description = "Green"  # This is where we specify "dog"

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

# vae.to("cpu")
# with torch.no_grad():
#     decoded_image = vae.decode(denoised_latents.to("cpu")).sample.to("cpu")


with torch.no_grad():
    decoded_image = vae.decode(denoised_latents).sample
# Post-process Image
decoded_image = (decoded_image.clamp(-1, 1) + 1) / 2
decoded_image = decoded_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
decoded_image = (decoded_image * 255).astype(np.uint8)
emoji_image = Image.fromarray(decoded_image)

# Display the image
emoji_image.show()

