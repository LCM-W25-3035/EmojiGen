import pandas as pd

# Path to the folder containing the Parquet files
folder_path = '../data/images/Alexator26Sticker/'

# List of known filenames
parquet_files = [f'{folder_path}{str(i).zfill(4)}.parquet' for i in range(6)]

# Read all Parquet files into a list of DataFrames
dataframes = [pd.read_parquet(file) for file in parquet_files]

# Concatenate all DataFrames into one
alexsticker_df = pd.concat(dataframes, ignore_index=True)

alexsticker_df.head()

from tqdm import tqdm
from PIL import Image
from io import BytesIO
import os

# Loop through the 'cartoonized_image' column in the DataFrame with tqdm
for idx, row in tqdm(alexsticker_df.iterrows(), total=len(alexsticker_df), desc="Processing Images"):
    # Get the byte data from the 'cartoonized_image' column
    image_bytes = row['cartoonized_image']['bytes']
    # Create a BytesIO object from the byte data
    image = Image.open(BytesIO(image_bytes))
    # Save the image as a PNG file
    image.save(f'{folder_path}cartoonized_image_{idx}.png')





