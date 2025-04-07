#!/usr/bin/env python
# coding: utf-8


import torch
from torch.utils.data import DataLoader, random_split
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm.auto import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler


# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True



# Data Preparation
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
df = pd.read_parquet('../data/processed_sticker_dataset.parquet')
dataset = EmojiDiffusionDataset(df)
train_size = int(0.9 * len(dataset))
train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)



# Load Model
model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
pipe.enable_attention_slicing()
pipe.unet.enable_gradient_checkpointing()



# Disable Safety Checker
if pipe.safety_checker:
    pipe.safety_checker = lambda images, clip_input, **kwargs: (images, [False] * len(images))

# LoRA Configuration
lora_config = LoraConfig(
    r=16, lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.1, bias="none"
)
pipe.unet = get_peft_model(pipe.unet, lora_config)
pipe.unet.print_trainable_parameters()



# Accelerator Setup
accelerator = Accelerator(mixed_precision='fp16', gradient_accumulation_steps=4)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-4)
pipe.unet, optimizer, train_loader = accelerator.prepare(pipe.unet, optimizer, train_loader)



print(df.columns)



# Training Loop
num_epochs = 50
training_losses, validation_losses = [], []
output_dir = "../evaluation/sticker_lora"
os.makedirs(output_dir, exist_ok=True)

scaler = torch.amp.GradScaler()

for epoch in range(num_epochs):
    pipe.unet.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16, non_blocking=True)
        prompts = batch["prompt"]
        
        text_inputs = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
        text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}

        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(**text_inputs).last_hidden_state
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

        noise = torch.randn_like(latents, device=device, dtype=torch.float16)
        timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        with torch.amp.autocast("cuda"):
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        accelerator.backward(loss)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    training_losses.append(avg_train_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Train Loss: {avg_train_loss:.6f}")

    # Validation Loop
    pipe.unet.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            pixel_values = batch["pixel_values"].to(device, dtype=torch.float16, non_blocking=True)
            prompts = batch["prompt"]
            
            text_inputs = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
            text_inputs = {k: v.to(device, non_blocking=True) for k, v in text_inputs.items()}

            encoder_hidden_states = pipe.text_encoder(**text_inputs).last_hidden_state
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

            noise = torch.randn_like(latents, device=device, dtype=torch.float16)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Val Loss: {avg_val_loss:.6f}")

    # Save Model Checkpoint
    checkpoint_dir = f"../evaluation/sticker_diffusion_qlora/checkpoint_epoch_{epoch+1}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    pipe.unet.save_pretrained(checkpoint_dir)


# Save Final Model
pipe.unet.save_pretrained("../evaluation/sticker_diffusion_qlora/final_model")



# Plot Training vs Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs+1), training_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs+1), validation_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "training_validation_loss.png"))
plt.close()

