import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Device Configuration
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using NVIDIA GPU (CUDA)')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using Mac GPU (MPS)')
else:
    device = torch.device('cpu')
    print('Using CPU')
    
# ------------------------------
# Configs
# ------------------------------
image_size = 256
clip_embedding_dim = 512
batch_size = 64
num_epochs = 1000
learning_rate = 1e-4
save_interval = int(num_epochs / 50)
save_dir = "../saved_models"
eval_dir = "../evaluation/emoji_diffusion"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)
fid_device = torch.device("cpu") if device.type == "mps" else device
fid_metric = FrechetInceptionDistance(feature=2048).to(fid_device)

# Load the dataset
df = pd.read_parquet('../data/processed_emoji_dataset.parquet')
df["combined_embedding"] = df["combined_embedding"].apply(lambda x: np.array(x, dtype=np.float32))

# Train and validation split
# train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

class EmojiDiffusionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_tensor = torch.load(self.df.iloc[idx]['image_path']).float()
        embedding = torch.tensor(self.df.iloc[idx]['combined_embedding'], dtype=torch.float32)
        embedding = embedding / embedding.norm()  # normalize CLIP embedding
        skintone = int(self.df.iloc[idx]['skintone'])
        prompt = self.df.iloc[idx]['prompt']
        return {
            "pixel_values": image_tensor,
            "embedding": embedding,
            "skintone": skintone,
            "prompt": prompt,
        }

# Create DataLoader
dataset = EmojiDiffusionDataset(df)
# Splitting data to training and testing sets
train_samples = int(round(len(dataset)*0.90))
train_set, val_set = random_split(dataset, [train_samples, len(dataset) - train_samples])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# ------------------------------
# Model Setup
# ------------------------------
model = UNet2DConditionModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    cross_attention_dim=528
).to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Skintone embedding
num_skintone_classes = 6  # 0–5 inclusive
skintone_embedding_dim = 16  # small vector size (you can tune this)

skintone_embedding_layer = nn.Embedding(num_skintone_classes, skintone_embedding_dim).to(device)

# LR Scheduler and Early Stopping
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
early_stopping_patience = 7
best_fid = float('inf')
epochs_without_improvement = 0
best_model_path = os.path.join(save_dir, "emoji_diffusion.pth")


# ------------------------------
# Sampling Function
# ------------------------------
@torch.no_grad()
def generate_emoji(embedding, skintone_ids, model, scheduler, skintone_embedding_layer, num_steps=1000):
    model.eval()
    scheduler.set_timesteps(num_steps)

    bsz = embedding.size(0)

    # Normalize CLIP embeddings if not already normalized
    embedding = embedding / embedding.norm(dim=1, keepdim=True)

    # Get skintone embedding and concatenate
    skintone_embed = skintone_embedding_layer(skintone_ids)
    final_embedding = torch.cat([embedding, skintone_embed], dim=1).unsqueeze(1)  # (B, 1, 528)

    # Start from random noise
    image = torch.randn((bsz, 3, image_size, image_size)).to(device)

    for t in scheduler.timesteps:
        noise_pred = model(image, t, encoder_hidden_states=final_embedding).sample
        image = scheduler.step(noise_pred, t, image).prev_sample

    # Normalize to [0, 1]
    image = (image.clamp(-1, 1) + 1) / 2
    return image


# ------------------------------
# Training Loop
# ------------------------------
train_losses = []
val_losses = []
eval_epochs = []

model.train()
noise_scheduler.set_timesteps(1000)

for epoch in range(1, num_epochs + 1):
    epoch_loss = 0
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        clean_images = batch["pixel_values"].to(device)
        embeddings = batch["embedding"].to(device)
        skintone_ids = batch["skintone"].to(device)
        skintone_embeds = skintone_embedding_layer(skintone_ids)
        
        # Combine CLIP and skintone
        final_embedding = torch.cat([embeddings, skintone_embeds], dim=1)
        final_embedding = final_embedding.unsqueeze(1)
        
        # Fix for CLIP embedding shape
        # embeddings = embeddings.unsqueeze(1)  # Shape: (batch_size, 1, 512)

        noise = torch.randn_like(clean_images).to(device)
        timesteps = torch.randint(0, 1000, (clean_images.shape[0],), device=device).long()
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, encoder_hidden_states=final_embedding).sample
        loss = nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f}")

    
    # ------------------------------
    # Evaluation and Saving
    # ------------------------------
    if epoch == 1 or epoch % save_interval == 0 or epoch == num_epochs:
        model.eval()
        val_loss = 0
        eval_epochs.append(epoch)
        with torch.no_grad():
            for val_batch in val_loader:
                val_clean = val_batch["pixel_values"].to(device)
                val_embed = val_batch["embedding"].to(device)
                val_skintone_ids = val_batch["skintone"].to(device)
                val_skintone_embed = skintone_embedding_layer(val_skintone_ids)
                val_embed = torch.cat([val_embed, val_skintone_embed], dim=1).unsqueeze(1)
                val_noise = torch.randn_like(val_clean)
                val_t = torch.randint(0, 1000, (val_clean.shape[0],), device=device).long()
                val_noisy = noise_scheduler.add_noise(val_clean, val_noise, val_t)

                val_pred = model(val_noisy, val_t, encoder_hidden_states=val_embed).sample
                val_loss += nn.functional.mse_loss(val_pred, val_noise).item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch} | Val Loss: {avg_val_loss:.4f}")

        # Generate samples for FID
        def get_first_full_batch(loader, min_batch_size=64):
            for batch in loader:
                if batch["pixel_values"].size(0) >= min_batch_size:
                    return batch
            raise ValueError("No batch with enough samples found.")
        val_batch = get_first_full_batch(val_loader)  # For FID + sample gen only
        clip_embed = val_batch["embedding"][:64].to(device)
        skintone_ids = val_batch["skintone"][:64].to(device)
        skintone_embed = skintone_embedding_layer(skintone_ids)
        final_embed = torch.cat([clip_embed, skintone_embed], dim=1)
        
        val_generated = generate_emoji(
            embedding=clip_embed,
            skintone_ids=skintone_ids,
            model=model,
            scheduler=noise_scheduler,
            skintone_embedding_layer=skintone_embedding_layer
        )

        real_images = val_batch["pixel_values"][:64].to(device)
        real_images_uint8 = (real_images * 255).clamp(0, 255).to(torch.uint8).to(fid_device)
        val_generated_uint8 = (val_generated * 255).clamp(0, 255).to(torch.uint8).to(fid_device)

        fid_metric.reset()
        fid_metric.update(real_images_uint8, real=True)
        fid_metric.update(val_generated_uint8, real=False)
        fid_score = fid_metric.compute().item()
        print(f"Epoch {epoch} | FID Score: {fid_score:.4f}")

        # Save FID score
        with open(os.path.join(eval_dir, "fid_scores.txt"), "a") as f:
            f.write(f"Epoch {epoch}: FID = {fid_score:.4f}\n")

        # Save generated images
        save_image(val_generated, os.path.join(eval_dir, f"val_epoch_{epoch}.png"), nrow=8)

        # Save loss graph
        plt.figure(figsize=(10, 5))
        plt.plot(eval_epochs, train_losses[-len(eval_epochs):], label='Train Loss')
        plt.plot(eval_epochs, val_losses, label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(eval_dir, f"loss_epoch_{epoch}.png"))
        plt.close()

        # Adjust learning rate
        scheduler.step(fid_score)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.6f}")

        # Save best model
        if fid_score < best_fid:
            print(f"New best FID: {fid_score:.4f} (previous: {best_fid:.4f}) — saving model.")
            best_fid = fid_score
            torch.save(model.state_dict(), best_model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement in FID. {epochs_without_improvement} epochs without improvement.")

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break


fid_epochs = []
fid_values = []
with open(os.path.join(eval_dir, "fid_scores.txt")) as f:
    for line in f:
        parts = line.strip().split(": FID = ")
        fid_epochs.append(int(parts[0].split(" ")[1]))
        fid_values.append(float(parts[1]))

plt.figure(figsize=(10, 5))
plt.plot(fid_epochs, fid_values, label='FID Score')
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID Score Over Epochs")
plt.legend()
plt.grid()
plt.savefig(os.path.join(eval_dir, "fid_progression.png"))
plt.close()


import torchvision.transforms as T

model.eval()

# Prepare inverse transform for displaying
to_pil = T.ToPILImage()

# Prepare figure
num_samples = min(len(val_set), 64)
val_loader_vis = DataLoader(val_set, batch_size=num_samples, shuffle=False)
val_batch = next(iter(val_loader_vis))
val_embeddings = val_batch["embedding"].to(device)
val_skintone_ids = val_batch["skintone"].to(device)
val_skintone_embed = skintone_embedding_layer(val_skintone_ids)
val_final_embed = torch.cat([val_embeddings, val_skintone_embed], dim=1)
val_images = val_batch["pixel_values"].to(device)

# Generate emojis
# generated_images = generate_emoji(val_embeddings, model, noise_scheduler)
generated_images = generate_emoji(
            val_embeddings,
            skintone_ids=val_skintone_ids,
            model=model,
            scheduler=noise_scheduler,
            skintone_embedding_layer=skintone_embedding_layer
        )

# Convert and plot
fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(9, num_samples * 1.5))
fig.suptitle("Prompt | Real Emoji | Generated Emoji", fontsize=14)

for i in range(num_samples):
    prompt_text = val_set[i]["prompt"] if "prompt" in df.columns else "Prompt not available"
    
    real_img = val_images[i].cpu().clamp(0, 1)
    gen_img = generated_images[i].cpu().clamp(0, 1)

    axes[i, 0].text(0.5, 0.5, prompt_text, ha='center', va='center', wrap=True)
    axes[i, 0].axis('off')

    axes[i, 1].imshow(to_pil(real_img))
    axes[i, 1].axis('off')

    axes[i, 2].imshow(to_pil(gen_img))
    axes[i, 2].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.98)
plt.savefig(os.path.join(eval_dir, "emoji_comparison_grid.png"))
print("Generated emoji comparison chart and saved as 'emoji_comparison_grid.png'")


# ## Q-LoRA (Best Configurations - Increase Epoch)


import torch
from torch.utils.data import DataLoader, random_split
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt  # For plotting the loss graph

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Data Preparation: Only return image and prompt
class EmojiDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_tensor = torch.load(self.df.iloc[idx]['image_path']).float()
        prompt = self.df.iloc[idx]['prompt']
        return {"pixel_values": image_tensor, "prompt": prompt}

# Load Dataset
df = pd.read_parquet('../data/processed_emoji_dataset.parquet')
dataset = EmojiDiffusionDataset(df)
train_samples = int(len(dataset) * 0.9)
train_set, val_set = random_split(dataset, [train_samples, len(dataset) - train_samples])
# Using the full dataset for training; you can switch to train_set if preferred.
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1)

# Load Model
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32  # Important for compatibility with MPS
).to("mps")
pipe.enable_attention_slicing()  # Memory-efficient optimization

# Disable safety checker by using a dummy function that returns the images as-is.
if pipe.safety_checker is not None:
    pipe.safety_checker = lambda images, clip_input, **kwargs: (images, [False] * len(images))

# Q-LoRA configuration for UNet
lora_config_unet = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none"
)
pipe.unet = get_peft_model(pipe.unet, lora_config_unet)
pipe.unet.print_trainable_parameters()

# Accelerator setup without mixed precision (required for MPS)
accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=1)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)
pipe.unet, optimizer, train_loader = accelerator.prepare(pipe.unet, optimizer, train_loader)

# Lists to store loss values per epoch
training_losses = []
validation_losses = []

# Directory for saving generated images and loss graph
gen_output_dir = "../evaluation/lora"
os.makedirs(gen_output_dir, exist_ok=True)

# Increase epochs to 20
num_epochs = 16

for epoch in range(num_epochs):
    pipe.unet.train()
    total_train_loss = 0

    # ------------------------
    # Training Loop
    # ------------------------
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        optimizer.zero_grad()

        # Get pixel values and prompt text from the batch
        pixel_values = batch["pixel_values"].to("mps")
        prompts = batch["prompt"]

        # Tokenize the prompts and move tensors to the proper device
        text_inputs = pipe.tokenizer(
            prompts,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt"
        )
        text_inputs = {k: v.to("mps") for k, v in text_inputs.items()}

        # Get conditioning embeddings using the built-in text encoder
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(**text_inputs).last_hidden_state

        # Encode images to latent space
        with torch.no_grad():
            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

        # Generate random noise and timesteps
        noise = torch.randn_like(latents).to("mps")
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device="mps").long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        # Predict noise using the UNet with encoder_hidden_states from the text encoder
        noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

        # Compute loss and update
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        accelerator.backward(loss)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    training_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Training Loss: {avg_train_loss:.6f}")

    # ------------------------
    # Validation Loop (Loss Computation)
    # ------------------------
    pipe.unet.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation Loss"):
            pixel_values = batch["pixel_values"].to("mps")
            prompts = batch["prompt"]

            text_inputs = pipe.tokenizer(
                prompts,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                return_tensors="pt"
            )
            text_inputs = {k: v.to("mps") for k, v in text_inputs.items()}

            encoder_hidden_states = pipe.text_encoder(**text_inputs).last_hidden_state

            latents = pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents).to("mps")
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device="mps").long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Average Validation Loss: {avg_val_loss:.6f}")

    # ------------------------
    # Generate an Emoji with prompt "grinning face"
    # ------------------------
    pipe.unet.eval()
    with torch.no_grad():
        generated = pipe(prompt="grinning face", num_inference_steps=50, guidance_scale=7.5, width=256, height=256)
        image = generated.images[0]
        filename = os.path.join(gen_output_dir, f"epoch{epoch+1}_grinning_face.png")
        image.save(filename)
        print(f"Generated emoji saved to {filename}")

    # ------------------------
    # Save Checkpoint at End of Epoch
    # ------------------------
    checkpoint_dir = f"../evaluation/emoji_diffusion_qlora/checkpoint_epoch_{epoch+1}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    pipe.unet.save_pretrained(checkpoint_dir)

# Save final model
final_model_dir = "../evaluation/emoji_diffusion_qlora/final_model"
os.makedirs(final_model_dir, exist_ok=True)
pipe.unet.save_pretrained(final_model_dir)

# ------------------------
# Plot and Save Training vs Validation Loss Graph
# ------------------------
epochs = range(1, num_epochs + 1)
plt.figure(figsize=(8, 6))
plt.plot(epochs, training_losses, label='Training Loss', marker='o')
plt.plot(epochs, validation_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
loss_plot_path = os.path.join(gen_output_dir, "training_validation_loss.png")
plt.savefig(loss_plot_path)
plt.close()
print(f"Training and validation loss graph saved to {loss_plot_path}")


# ## Latest with LoRA on text encoder as well

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel, LoraConfig
from PIL import Image
import os

# Device configuration (use 'cuda' if available, otherwise 'mps' or 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

# Load the base pipeline
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
).to(device)
pipe.enable_attention_slicing()  # Memory-efficient optimization

# Disable the safety checker to bypass NSFW filtering
if pipe.safety_checker is not None:
    pipe.safety_checker = lambda images, clip_input, **kwargs: (images, [False] * len(images))

# Q-LoRA configuration (must match your training configuration)
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1,
    bias="none"
)

# Directory for the final trained LoRA weights
final_model_dir = "../evaluation/emoji_diffusion_qlora/final_model"

# Load the LoRA weights onto the UNet using PeftModel.from_pretrained
pipe.unet = PeftModel.from_pretrained(pipe.unet, final_model_dir)

# Inference parameters
prompt = "monkey with banana"  # Change to your desired prompt
num_inference_steps = 50
guidance_scale = 7.5

# Generate the image
with torch.no_grad():
    generated = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=256, width=256)
    image = generated.images[0]

# Save the generated image
output_path = "cat_face.png"
os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) != "" else ".", exist_ok=True)
image.save(output_path)
print(f"Inference result saved to {output_path}")

# Optionally, display the image
image.show()



