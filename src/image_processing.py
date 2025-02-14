import os
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

image_base_path = "../data/images/"
tensor_output_path = "../data/tensor_images/"

os.makedirs(tensor_output_path, exist_ok=True)

emoji_categories = ["GoogleEmoji", "JoyPixelsEmoji", "OpenMojiEmoji", "TwitterEmoji"]
sticker_categories = ["AlexatorStickers", "FlaticonStickers", "FreepikStickers"]

emoji_base_transform = transforms.Compose([
    transforms.Resize((16, 16)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

sticker_base_transform = transforms.Compose([
    transforms.Resize((32, 32)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

for category in tqdm(emoji_categories, desc="Processing Emoji Categories"):
    category_folder = os.path.join(image_base_path, category)
    tensor_category_path = os.path.join(tensor_output_path, category)

    os.makedirs(tensor_category_path, exist_ok=True)

    for filename in os.listdir(category_folder):
        # Added this code to ignore the .DS_Store file on mac
        if filename == ".DS_Store":  # Ignore macOS system file
            continue
        
        img_path = os.path.join(category_folder, filename)

        new_name = filename.replace("emoji_u", "").replace("_", "-").lower()
        new_name = new_name.replace("-fe0f", "").replace("-200d", "")
        
        img = Image.open(img_path) # Path of the image
        
        if img.mode == "P":  # Fix transparency warning by converting png files to RGBA first
                img = img.convert("RGBA")

        img = img.convert("RGB")  # Convert to RGB format
        img_tensor = sticker_base_transform(img)

        # Save as .pt file
        tensor_file = new_name.replace(".png", ".pt").replace(".jpg", ".pt")
        torch.save(img_tensor, os.path.join(tensor_category_path, tensor_file))

        # print(f"Processed: {filename} -> {tensor_file}")

print("All emoji images processed successfully!")

for category in tqdm(emoji_categories, desc="Processing Sticker Categories"):
    category_folder = os.path.join(image_base_path, category)
    tensor_category_path = os.path.join(tensor_output_path, category)

    os.makedirs(tensor_category_path, exist_ok=True)

    for filename in os.listdir(category_folder):
        # Added this code to ignore the .DS_Store file on mac
        if filename == ".DS_Store":  # Ignore macOS system file
            continue
        
        img_path = os.path.join(category_folder, filename)
        
        img = Image.open(img_path) # Path of the image
        
        if img.mode == "P":  # Fix transparency warning by converting png files to RGBA first
                img = img.convert("RGBA")

        img = img.convert("RGB")  # Convert to RGB format
        img_tensor = emoji_base_transform(img)

        # Save as .pt file
        tensor_file = filename.replace(".png", ".pt").replace(".jpg", ".pt")
        torch.save(img_tensor, os.path.join(tensor_category_path, tensor_file))

        # print(f"Processed: {filename} -> {tensor_file}")

print("All sticker images processed successfully!")

# Prompt(GPT 4o): make data loader to load above image file which are saved in tensor
class EmojiDataset(Dataset):
    def __init__(self, tensor_dir, transform=None):
        self.tensor_dir = tensor_dir
        self.transform = transform
        self.tensor_files = [os.path.join(root, file) 
                             for root, _, files in os.walk(tensor_dir) 
                             for file in files if file.endswith('.pt')]

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_path = self.tensor_files[idx]
        image_tensor = torch.load(tensor_path)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor

# Create dataset and dataloader
emoji_dataset = EmojiDataset(tensor_output_path)
emoji_dataloader = DataLoader(emoji_dataset, batch_size=32, shuffle=True)

# Example: Iterate through the dataloader
for batch in emoji_dataloader:
    print(batch.shape)
    break