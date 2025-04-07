#!/usr/bin/env python
# coding: utf-8



import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from tqdm import tqdm




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Load the dataset
#Load the sticker dataset
df = pd.read_parquet('../data/processed_sticker_dataset.parquet')

# Convert embeddings to float32 numpy array
df["combined_embedding"] = df["combined_embedding"].apply(lambda x: np.array(x, dtype=np.float32))




class StickerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        embedding = torch.tensor(self.dataframe.iloc[idx]['combined_embedding']).float()
        image_tensor = torch.load(self.dataframe.iloc[idx]['image_path']).float()
        return embedding, image_tensor




dataset = StickerDataset(df)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)




class Generator(nn.Module):
    def __init__(self, noise_dim, embedding_dim, image_channels=3):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_channels * 32 * 32),
            nn.Tanh()
        )

    def forward(self, noise, embed):
        x = torch.cat((noise, embed), dim=1)
        x = self.model(x)
        return x.view(x.size(0), 3, 32, 32)




class Discriminator(nn.Module):
    def __init__(self, embedding_dim, image_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_channels * 32 * 32 + embedding_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, embed):
        img_flat = img.view(img.size(0), -1)
        x = torch.cat((img_flat, embed), dim=1)
        return self.model(x)




noise_dim = 100
embedding_dim = len(df['combined_embedding'][0])
generator = Generator(noise_dim, embedding_dim).to(device)
discriminator = Discriminator(embedding_dim).to(device)

criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

num_epochs = 50

for epoch in range(num_epochs):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
    
    for embeddings, real_images in progress_bar:
        embeddings, real_images = embeddings.to(device), real_images.to(device)
        
        # Train Discriminator
        noise = torch.randn(real_images.size(0), noise_dim).to(device)
        fake_images = generator(noise, embeddings)
        
        real_labels = torch.ones(real_images.size(0), 1).to(device)
        fake_labels = torch.zeros(real_images.size(0), 1).to(device)
        
        real_outputs = discriminator(real_images, embeddings)
        fake_outputs = discriminator(fake_images.detach(), embeddings)
        
        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        fake_outputs = discriminator(fake_images, embeddings)
        g_loss = criterion(fake_outputs, real_labels)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")




import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

def generate_stickers(descriptions, generator):
    generator.eval()
    stickers = []
    for description in descriptions:
        embedding = sbert_model.encode(description)
        embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        noise = torch.randn(1, noise_dim).to(device)
        with torch.no_grad():
            fake_image = generator(noise, embedding).squeeze(0).cpu().numpy()
        stickers.append(fake_image)
    return stickers

def display_stickers(images):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 3, 3))
    for ax, image in zip(axes, images):
        image = (image + 1) / 2  # Normalize to [0,1]
        ax.imshow(np.transpose(image, (1, 2, 0)))
        ax.axis("off")
    plt.show()

# Example: Generate & Display Multiple Stickers
descriptions = ["A happy sun wearing sunglasses", "A cute panda waving", "A colorful unicorn smiling"]
stickers = generate_stickers(descriptions, generator)
display_stickers(stickers)

