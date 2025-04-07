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

# Load the dataset
df = pd.read_parquet('../data/processed_emoji_dataset.parquet')
df["combined_embedding"] = df["combined_embedding"].apply(lambda x: np.array(x, dtype=np.float32))

# Custom Dataset Class
class EmojiDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        embedding = torch.tensor(self.dataframe.iloc[idx]['combined_embedding']).float()
        image_tensor = torch.load(self.dataframe.iloc[idx]['image_path']).float()
        return embedding, image_tensor

# Create DataLoader
dataset = EmojiDataset(df)
# Splitting data to training and testing sets
train_samples = int(round(len(dataset)*0.90))
train_set, val_set = random_split(dataset, [train_samples, len(dataset) - train_samples])
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# Directories to save
os.makedirs("../evaluation/emoji_cgan/train_output", exist_ok=True)
os.makedirs("../evaluation/emoji_cgan/val_output", exist_ok=True)
os.makedirs("../saved_models", exist_ok=True)
train_output_dir = "../evaluation/emoji_cgan/train_output"
val_output_dir = "../evaluation/emoji_cgan/val_output"
val_models_dir = "../saved_models"

"""
Reference: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
Learning ConvTranspose2d layers for upsampling
"""
# Generator Model
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

"""
Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
Learning Conv2d layers for downsampling
"""
# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self, embedding_dim, image_channels=3):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(image_channels + 1, 64, kernel_size=4, stride=2, padding=1),  # added +1
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),  # Dropout to weaken discriminator
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),  # Dropout to weaken discriminator
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),  # Dropout to weaken discriminator
        )
        self.fc = nn.Linear(256 * 4 * 4, 1)
        # Project embedding into a spatial format
        self.embed_fc = nn.Linear(embedding_dim, 4 * 4)  # Map embeddings to 4x4 spatial size
        
    def forward(self, img, embed):
        batch_size = img.size(0)
        # Convert embedding into spatial form
        embed_features = self.embed_fc(embed).view(batch_size, 1, 4, 4)
        # Resize embedding map to match image dimensions
        embed_features = torch.nn.functional.interpolate(embed_features, size=(img.shape[2], img.shape[3]))
        # Concatenate embeddings as an extra channel
        x = torch.cat((img, embed_features), dim=1)
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        return torch.sigmoid(self.fc(x))

# Model Initialization
noise_dim = 100
embedding_dim = len(df['combined_embedding'][0])
generator = Generator(noise_dim, embedding_dim).to(device)
discriminator = Discriminator(embedding_dim).to(device)
gamma = 10.0  # R1 regularization coefficient

patience = 5  # Number of epochs to wait before early stopping
lr_patience = 1  # Number of epochs to wait before reducing learning rate
lr_factor = 0.5  # Factor to reduce learning rate by
min_lr = 1e-6  # Minimum learning rate threshold

best_fid_score = float('inf')
epochs_since_improvement = 0
epochs_since_lr_reduce = 0

# Loss and Optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Prepare InceptionV3 model for FID calculation using the new weights API.
inception_model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                        transform_input=False,
                                        aux_logits=True).to(device)
# Replace the final fully connected layer with an identity so that we get features
inception_model.fc = nn.Identity()
inception_model.eval()

def get_inception_features(images, model):
    """
    Resizes images to 299x299, normalizes them with Inception's mean and std,
    and returns the features from the model.
    """
    # Resize to InceptionV3 expected input size
    images = torch.nn.functional.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    # If images are from generator (range [-1, 1]), convert them to [0, 1]
    images = (images + 1) / 2
    # Normalize with ImageNet statistics
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    images = torch.stack([normalize(img) for img in images])
    with torch.no_grad():
        features = model(images.to(device))
        # If model returns a tuple (due to aux logits), take the first element.
        if isinstance(features, tuple):
            features = features[0]
        features = features.detach().cpu().numpy()
    return features

def compute_fid(real_images, generated_images, model):
    """
    Computes the Frechet Inception Distance (FID) between two sets of images.
    """
    # Get inception features for real and generated images
    real_features = get_inception_features(real_images, model)
    fake_features = get_inception_features(generated_images, model)
    
    # Compute mean and covariance statistics
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Compute squared difference between means
    diff = mu_real - mu_fake
    diff_squared = diff.dot(diff)
    
    # Compute sqrt of product of covariance matrices
    covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
    # If the product is almost singular, sqrtm may return complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff_squared + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

num_epochs = 5000
save_interval = int(num_epochs / 50)
d_losses, g_losses, d_fake_losses, d_real_losses = [], [], [], []
fid_scores = []
fid_epochs = []
all_fid_scores = []
all_generated_images = []
all_real_images = []
all_epochs = []

for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    for i, (combined_embeddings, real_images) in enumerate(train_loader):
        combined_embeddings = combined_embeddings.to(device)
        real_images = real_images.to(device)
        # Ensure real images require gradients for R1 penalty computation.
        real_images.requires_grad_()
        
        # Train Discriminator
        noise = torch.randn(real_images.size(0), noise_dim).to(device)
        fake_images = generator(noise, combined_embeddings)
        
        real_labels = torch.full((real_images.size(0), 1), 0.95).to(device)
        fake_labels = torch.full((real_images.size(0), 1), 0.05).to(device)
        
        # Forward pass on real images.
        real_outputs = discriminator(real_images, combined_embeddings)
        d_loss_real = criterion(real_outputs, real_labels)
        
        # Compute R1 regularization: gradient penalty on real images.
        grad_real = torch.autograd.grad(
            outputs=real_outputs.sum(), 
            inputs=real_images, 
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_penalty = grad_real.view(grad_real.size(0), -1).pow(2).sum(1).mean()
        d_loss_real = d_loss_real + (gamma / 2) * grad_penalty
        
        # Forward pass on fake images.
        fake_outputs = discriminator(fake_images.detach(), combined_embeddings)
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # Train Generator
        fake_outputs = discriminator(fake_images, combined_embeddings)
        g_loss = criterion(fake_outputs, real_labels)
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
        
        # Turn off gradients for real_images after update.
        real_images.requires_grad_(False)
    
    # Store loss values
    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())
    d_real_losses.append(d_loss_real.item())
    d_fake_losses.append(d_loss_fake.item())
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss:.4f} | G Loss: {g_loss.item():.4f}")
    
    # Save evaluation every save_interval
    if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(val_output_dir, exist_ok=True)

        generator.eval()

        # --- Evaluate on Training Set ---
        train_batch = next(iter(train_loader))
        train_embeddings, train_images = train_batch
        train_embeddings = train_embeddings.to(device)
        noise_train = torch.randn(train_images.size(0), noise_dim).to(device)
        with torch.no_grad():
            generated_train = generator(noise_train, train_embeddings).cpu()
        grid_train = make_grid(generated_train, nrow=8, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid_train.numpy(), (1, 2, 0)))
        plt.title(f"Train Set Generated Images at Epoch {epoch+1}")
        plt.axis("off")
        train_image_path = os.path.join(train_output_dir, f"generated_train_epoch_{epoch+1}.png")
        plt.savefig(train_image_path)
        plt.close()

        # --- Plot Loss Curves ---
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(g_losses) + 1), g_losses, label="G Loss")
        plt.plot(range(1, len(d_losses) + 1), d_losses, label="D Loss")
        plt.xlabel("Epoch (save interval count)")
        plt.ylabel("Loss")
        plt.title(f"Training Losses up to Epoch {epoch+1}")
        plt.legend()
        loss_plot_path = os.path.join(train_output_dir, f"loss_plot_epoch_{epoch+1}.png")
        plt.savefig(loss_plot_path)
        plt.close()

        # --- Evaluate on Validation Set ---
        val_batch = next(iter(val_loader))
        val_embeddings, val_images = val_batch
        val_embeddings = val_embeddings.to(device)
        noise_val = torch.randn(val_images.size(0), noise_dim).to(device)
        with torch.no_grad():
            generated_val = generator(noise_val, val_embeddings).cpu()
        grid_val = make_grid(generated_val, nrow=8, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(grid_val.numpy(), (1, 2, 0)))
        plt.title(f"Validation Set Generated Images at Epoch {epoch+1}")
        plt.axis("off")
        val_image_path = os.path.join(val_output_dir, f"generated_val_epoch_{epoch+1}.png")
        plt.savefig(val_image_path)
        plt.close()

        # --- Compute FID Score on Validation Set ---
        # Note: using the current batch from the validation set for demonstration.
        fid_score = compute_fid(val_images.cpu(), generated_val, inception_model)
        fid_scores.append(fid_score)
        fid_epochs.append(epoch+1)
        print(f"Epoch {epoch+1} FID Score: {fid_score:.4f}")

        all_fid_scores.append(fid_score)
        all_generated_images.append(generated_val.cpu())
        all_real_images.append(val_images.cpu())
        all_epochs.append(epoch + 1)
        
        # Early Stopping and LR Reduction Logic
        if fid_score < best_fid_score and epoch > 2000:
            best_fid_score = fid_score
            epochs_since_improvement = 0
            epochs_since_lr_reduce = 0
    
            # Save the best models
            torch.save(generator.state_dict(), os.path.join(val_models_dir, "cgan_emoji_generator.pth"))
            torch.save(discriminator.state_dict(), os.path.join(val_models_dir, "cgan_emoji_discriminator.pth"))
            print(f"Saved improved model at Epoch {epoch+1} with FID {fid_score:.4f}")
        else:
            epochs_since_improvement += 1
            epochs_since_lr_reduce += 1
            
        # Reduce learning rate if no improvement for 'lr_patience' epochs
        if epochs_since_lr_reduce >= lr_patience and epoch > 2000:
            new_g_lr = max(g_optimizer.param_groups[0]['lr'] * lr_factor, min_lr)
            new_d_lr = max(d_optimizer.param_groups[0]['lr'] * lr_factor, min_lr)
    
            for param_group in g_optimizer.param_groups:
                param_group['lr'] = new_g_lr
            for param_group in d_optimizer.param_groups:
                param_group['lr'] = new_d_lr
    
            print(f"Reducing learning rates to Generator: {new_g_lr}, Discriminator: {new_d_lr}")
    
            epochs_since_lr_reduce = 0
    
        # Early stopping if no improvement for 'patience' epochs
        if epochs_since_improvement >= patience and epoch > 2000:
            print(f"No improvement in FID for {patience} intervals. Stopping early at Epoch {epoch+1}.")
            break

        generator.train()

#Plot the FID scores over epochs
plt.figure(figsize=(10, 5))
plt.plot(fid_epochs, fid_scores, marker='o', label="FID Score")
plt.xlabel("Epoch")
plt.ylabel("FID Score")
plt.title("FID Score Over Training")
plt.legend()
fid_plot_path = os.path.join(val_output_dir, "fid_score_plot.png")
plt.savefig(fid_plot_path)
plt.close()
print("Training complete. Best model saved with FID:", best_fid_score)


# ## Generating emojis from prompts

"""
Reference: ChatGPT-4.5
Prompt: I want to generate emojis using prompts. Input 5 different prompts, and save a plot that displays the prompts and their corresponding emoji generated using my generator. I'm using CLIP embedding for my texts.
"""

# Define your prompts
prompts = [
    "Crying face",
    "A brown man running in the left direction",
    "An angry red face",
    "A face wearing sunglasses depicting a sense of coolness",
    "A man and a woman in love"
]

generated_images = []

from transformers import CLIPTokenizer, CLIPTextModel

# Load CLIP's tokenizer and text model.
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
    if pd.isna(text) or text.strip() == "":
        # Adjust the zero vector size to match CLIP's output dimension (512 for clip-vit-base-patch32)
        return np.zeros(512, dtype=np.float32)
    
    # Tokenize the input text
    inputs = clip_tokenizer(text, return_tensors="pt", truncation=True, max_length=77)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Disable gradients for inference
    with torch.no_grad():
        output = clip_model(**inputs)
    
    # Pool the token embeddings (mean pooling)
    pooled_embedding = mean_pooling(output, inputs["attention_mask"])
    
    # Optionally, you might want to L2 normalize the pooled embedding:
    pooled_embedding = torch.nn.functional.normalize(pooled_embedding, p=2, dim=-1)
    
    return pooled_embedding.squeeze().cpu().numpy().astype(np.float32)


# Generate an emoji for each prompt using the Hugging Face CLIP resources
for prompt in prompts:
    # Tokenize the prompt using the CLIPTokenizer and get the embedding as a numpy array
    text_embedding = embed_text(prompt)
    # Convert the numpy array back to a torch tensor and add the batch dimension
    text_embedding = torch.tensor(text_embedding).to(device).unsqueeze(0)
    
    # Generate a random noise vector
    noise = torch.randn(1, noise_dim).to(device)
    
    # Generate the emoji image using your generator
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
save_path = os.path.join(val_output_dir, "emojis_from_prompts.png")
plt.savefig(save_path)
print(f"Saved generated emojis plot at: {save_path}")
plt.close()




