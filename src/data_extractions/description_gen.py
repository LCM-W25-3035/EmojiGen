# 🚀 Prompt Summary
# First Prompt: Generate image captions using HuggingFace image captioning models from local image folders.
# Last Prompt: Cycle through multiple HuggingFace API keys for robustness and update sticker descriptions in CSV.
# Model Used: ChatGPT (gpt-4-turbo)
# -----------------------------------------------

import pandas as pd

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import requests
import time
from tqdm import tqdm

# 🔹 Hugging Face API Configuration
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
API_KEYS= ["USE your API keys as list"]
current_api_index = 0  # Start with the first API key
used_api_keys = set()  # Keep track of used API keys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folder = os.path.join(PROJECT_ROOT, "data")
flaticon = os.path.join(data_folder,"flaticon")
freepik = os.path.join(data_folder,"freepik")
alexator = os.path.join(data_folder,"alexator")

flaticon_images= os.path.join(flaticon,"images")
freepik_images= os.path.join(freepik,"images")
alexator_images= os.path.join(alexator,"images")

flaticon_csv= os.path.join(flaticon,"flaticon_stickers_desc.csv")
freepik_csv= os.path.join(freepik,"freepik_stickers_desc.csv")
alexator_csv = os.path.join(alexator,"alexator_stickers_desc.csv")
# 🔹 Data Folders and CSV Files
DATA_FOLDERS = {
    "flaticon": {"image_dir": flaticon_images, "csv_file": flaticon_csv},
    "freepik": {"image_dir": freepik_images, "csv_file": freepik_csv},
    "alexator": {"image_dir": alexator_images, "csv_file": alexator_csv},  # No CSV, needs to be created
}

# 🔹 Function to Call Hugging Face API
def generate_caption(image_path):
    """Generate a description using Hugging Face API, switching API keys if limit is reached or key is invalid."""
    global current_api_index

    while current_api_index < len(API_KEYS):
        headers = {"Authorization": f"Bearer {API_KEYS[current_api_index]}"}
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            response = requests.post(API_URL, headers=headers, data=image_bytes)
            result = response.json()

            # ✅ If a valid response, return generated caption
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "No caption generated")

            # ⚠️ Handle API rate limit error, switch to next key
            elif "error" in result:
                error_msg = result["error"].lower()
                if "limit" in error_msg:
                    logging.info(f"\n⚠️ API limit reached for key {current_api_index + 1}. Switching to next key...")
                elif "unauthorized" in error_msg or "invalid" in error_msg:
                    logging.info(f"\n❌ Invalid or expired API key detected: {API_KEYS[current_api_index]}. Skipping...")
                elif "unavailable" or "currently loading" in error_msg:
                    continue
                else:
                    logging.info(error_msg)
                # Mark key as used and move to the next one
                used_api_keys.add(API_KEYS[current_api_index])
                current_api_index += 1
                logging.info(current_api_index)
                time.sleep(1)  # Short delay before retrying
                continue  # Retry with the next API key

            else:
                return f"API Error: {result}"

        except Exception as e:
            return f"Error processing image: {str(e)}"

    return "❌ No working API keys available!"

# 🔹 Process Each Data Folder
for source, paths in DATA_FOLDERS.items():
    image_dir = paths["image_dir"]
    csv_file = paths["csv_file"]
    output_csv = f"{data_folder}/{source}/{source}_stickers_desc.csv"  # Save descriptions here

    # Load existing data if CSV exists, else create a new DataFrame
    if csv_file and os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        if "description" not in df.columns:
            df["description"] = ""
    else:
        filenames = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        df = pd.DataFrame({"filename": filenames, "description": ""})  # Create new CSV for alexator

    # Ensure we only process images that don't have a description
    descriptions = []

    for i, filename in tqdm(enumerate(df["filename"]), desc=f"Processing {source}", unit="image", total=len(df)):
        image_path = os.path.join(image_dir, filename)

        # Skip images that already have a description
        if pd.notna(df.loc[i, "description"]) and (df.loc[i, "description"] != "" or df.loc[i,"description"]!="❌ No working API keys available!"):
            descriptions.append(df.loc[i, "description"])
            continue

        # Generate a new caption
        description = generate_caption(image_path)
        if description == "❌ No working API keys available!":
            continue
        descriptions.append(description)
        df.loc[i, "description"] = description  # Store in DataFrame

        # Save progress every 10 images
        if (i + 1) % 500 == 0:
            df.to_csv(output_csv, index=False)

    # Final save
    df.to_csv(output_csv, index=False)
    logging.info(f"\n✅ Finished processing {source}. Updated CSV saved as {output_csv}")

logging.info("\n🎉 All datasets processed successfully!")