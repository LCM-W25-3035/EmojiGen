import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm
import torch
# from sentence_transformers import SentenceTransformer
# from spellchecker import SpellChecker
import string
import matplotlib.pyplot as plt
from collections import Counter
# Apply tqdm to all .apply() functions by using progress_apply
tqdm.pandas()
# Use GPU if available
"""
Reference: https://pytorch.org/get-started/locally/
"""

# Check for NVIDIA GPU
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA (NVIDIA GPU)
    print("Using NVIDIA GPU (CUDA)")

# Check for Mac Silicon GPU (MPS)
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use Metal Performance Shaders (Mac Silicon GPU)
    print("Using Mac GPU (MPS)")

# Default to CPU if no GPU is available
else:
    device = torch.device("cpu")
    print("Using CPU")


# Reading the dataset
alexator_df = pd.read_csv('../data/alexator_stickers_desc.csv')


alexator_df.info()

# Concatenate the DataFrames row-wise
# merged_df = pd.concat([alexator_df, flaticon_df, freepik_df], ignore_index=True)
merged_df = alexator_df
# Display the concatenated DataFrame
merged_df.info()
final_df = merged_df.copy()
final_df.info()
final_df.head()
# Change file extensions from .png to .pt in the 'filename' column
final_df['filename'] = final_df['filename'].str.replace('.png', '.pt', regex=False)
# Display the first few rows to verify changes
final_df['filename'].head()
# Define the base directory and folders to search in
base_dir = "../data/tensor_images/"
folders = ["AlexatorStickers"]

# Function to find the file path for a given filename
def find_image_path(filename):
    # Loop through each folder
    for folder in folders:
        # Construct full path
        full_path = os.path.join(base_dir, folder, filename)
        # Check if file exists at this path
        if os.path.exists(full_path):
            return full_path
    # Return None if file not found in any folder
    return None

# Create the new column by applying the function to the filename column
final_df['image_path'] = final_df['filename'].progress_apply(find_image_path)

# Display a sample of the DataFrame to verify
final_df.head()
final_df.rename(columns={'description': 'prompt'}, inplace=True)
final_df.tail()
final_df.info()
final_df.to_parquet('../data/processed_sticker_dataset.parquet', index=False)

