import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# Download NLTK data files (only need to run once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

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
openmoji_df = pd.read_csv('../data/openmoji.csv')
llm_df = pd.read_parquet('../data/llmemoji.parquet')
emojipedia_df = pd.read_csv('../data/emojipedia.csv')




# Convert Unicode string (e.g., 'U+1F600', 'U+263A,FE0F') to hex code ('1F600', '263A-FE0F').

def unicode_to_hex(unicode_str):
    unicode_str = unicode_str.replace(",", " ")  # Replace commas with spaces
    # First splitting the input string to a list of substrings
    # Loops though each substring
    # Removes the U+ prefix from each substring
    hex_values = [u.replace("U+", "") for u in unicode_str.split()]
    # Join the values with hyphens
    return "-".join(hex_values)

# Convert 'unicode' column in emojipedia_df and llm_df to 'hexcode'
llm_df['hexcode'] = llm_df['unicode'].progress_apply(unicode_to_hex)
emojipedia_df['hexcode'] = emojipedia_df['Codepoints Hex'].progress_apply(unicode_to_hex)




# Making the hexcode uniform in all 3 dataframes for merging
# removing -f30f (differentiation between image type emoji and textual type emoji)
# removing -200d (differentiation for emoji with skin-tone)

openmoji_df['hexcode'] = openmoji_df['hexcode'].str.replace('-FE0F', '', regex=True)
openmoji_df['hexcode'] = openmoji_df['hexcode'].str.replace('-200D', '', regex=True)
openmoji_df['skintone_base_hexcode'] = openmoji_df['skintone_base_hexcode'].str.replace('-FE0F', '', regex=True)
openmoji_df['skintone_base_hexcode'] = openmoji_df['skintone_base_hexcode'].str.replace('-200D', '', regex=True)
llm_df['hexcode'] = llm_df['hexcode'].str.replace('-FE0F', '', regex=True)
llm_df['hexcode'] = llm_df['hexcode'].str.replace('-200D', '', regex=True)
emojipedia_df['hexcode'] = emojipedia_df['hexcode'].str.replace('-FE0F', '', regex=True)
emojipedia_df['hexcode'] = emojipedia_df['hexcode'].str.replace('-200D', '', regex=True)



openmoji_df['hexcode'] = openmoji_df['hexcode'].str.lower()
openmoji_df['skintone_base_hexcode'] = openmoji_df['skintone_base_hexcode'].str.lower()
llm_df['hexcode'] = llm_df['hexcode'].str.lower()
emojipedia_df['hexcode'] = emojipedia_df['hexcode'].str.lower()




# Dropping symbols, extras-openmoji, extras-unicode and flags categories
openmoji_df = openmoji_df[~openmoji_df["group"].isin(["symbols", "extras-openmoji", "extras-unicode", "flags"])]

# Dropping records of skin colors and hair types
openmoji_df = openmoji_df[~openmoji_df["hexcode"].isin(["1f3fb", "1f3fc", "1f3fd", "1f3fe", "1f3ff", "1f9b0", "1f9b1", "1f9b3", "1f9b2"])]

# Drop all records with multiple skintone combinations
openmoji_df = openmoji_df[~openmoji_df["skintone_combination"].isin(["multiple"])]

# Setting skintone to 0 where no skintone is specified
openmoji_df['skintone'] = openmoji_df['skintone'].fillna(0)




# Checking for duplicates
duplicate_counts = openmoji_df['hexcode'].value_counts()
duplicates = duplicate_counts[duplicate_counts > 1]
print(duplicates)




# Checking for duplicates
duplicate_counts = llm_df['hexcode'].value_counts()
duplicates = duplicate_counts[duplicate_counts > 1]
print(duplicates)




# Checking for duplicates
duplicate_counts = emojipedia_df['hexcode'].value_counts()
duplicates = duplicate_counts[duplicate_counts > 1]
print(duplicates)




# Removing duplicates
llm_df = llm_df[~llm_df.duplicated(subset=['hexcode'], keep=False)]




# Merge the dataframes on 'hexcode' with left join on openmoji_df
merged_df = openmoji_df.merge(llm_df, on='hexcode', how='left')
merged_df = merged_df.merge(emojipedia_df, on='hexcode', how='left')




merged_df.info()


# ## Handling Tags and Descriptions



def get_first_sentence(text):
    """
    Extracts the first sentence from a given text.
    A sentence ends with '.', '?', or '!' followed by space or end of string.
    """
    if not isinstance(text, str):
        return text  # Return as-is if not a string
    
    match = re.match(r'(.+?[.!?])(\s|$)', text.strip())
    return match.group(1) if match else text

merged_df["Description"] = merged_df["Description"].apply(get_first_sentence)




import unicodedata

def clean_text(text):
    """
    Normalize and remove unwanted characters.
    """
    if not isinstance(text, str):
        return text

    text = unicodedata.normalize("NFKD", text)
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"():;\-\n]+', '', text)




# def clean_text(text):
#     if not isinstance(text, str) or pd.isna(text) or text.strip().lower() == "nan":  
#         return ""  # Return empty string for NaN or "nan" strings
#     text = text.lower().strip() # Convert to lowercase and remove unnecessary spaces
#     # Keep only letters, numbers, spaces, * and #
#     text = re.sub(r'[^a-z0-9\s*#]', '', text)
#     # Tokenize the text
#     tokens = word_tokenize(text)
#     # Remove stop words
#     tokens = [word for word in tokens if word not in stopwords.words('english')]
#     return ', '.join(tokens)
# 
# def remove_duplicates(text):
#     words = [word.strip() for word in text.split(",")]  # Split by commas and strip spaces
#     unique_words = list(dict.fromkeys(words))
#     return ', '.join(unique_words)  # Join back into a string




# Cleaning annotation, LLM description and openmoji_description
merged_df["annotation"] = merged_df["annotation"].apply(clean_text)
merged_df["LLM description"] = merged_df["LLM description"].apply(clean_text)
merged_df["Description"] = merged_df["Description"].apply(clean_text)




# Create a mapping from hexcode to Description (only non-null ones)
hexcode_to_description = merged_df[merged_df['Description'].notnull()] \
    .set_index('hexcode')['Description'].to_dict()

# Fill missing Descriptions based on skintone_base_hexcode
merged_df['Description'] = merged_df.apply(
    lambda row: hexcode_to_description.get(row['skintone_base_hexcode'], row['Description'])
    if pd.isnull(row['Description']) else row['Description'],
    axis=1
)




# Merge annotation with a description
# Use LLM description if it exists, otherwise use Description
def generate_prompt(row):
    # Start with annotation (always exists)
    prompt = row['annotation']

    # Try LLM description, then Description
    extra = row['LLM description'] if pd.notnull(row['LLM description']) else row['Description']

    # If there's extra content, append with a space
    if pd.notnull(extra):
        prompt += '. ' + extra

    return prompt

merged_df['prompt'] = merged_df.apply(generate_prompt, axis=1)



merged_df.info()


# ## Embedding Emoji Condition



"""
Reference: https://huggingface.co/docs/transformers/model_doc/clip
"""

# Load CLIP's tokenizer and text model.
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model = clip_model.to(device)
clip_model.eval()




"""
Reference: ChatGPT o3-mini-high
Prompt: Write a code to embed my description column in my df using CLIP-vit-base-patch32. Use Mean Pooling + L2 Normalization method to generate embeddings.

Reason: We're using Mean Pooling + L2 Normalization to retain fine-grained meanings related to gender, skin tone, emotions, and objects. We're also using L2 Normalization because they have a consistent scale, reducing variance in GAN training.
"""

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




# Apply CLIP embedding to your dataset
merged_df["combined_embedding"] = merged_df["prompt"].progress_apply(embed_text)




merged_df.head()


# ## Linking Images



# Define base image path and brands
image_base_path = "../data/tensor_images/"
# brands = ["GoogleEmoji", "JoyPixelsEmoji", "TwitterEmoji"]
brands = ["OpenMojiEmoji"]

# Function to find all available image paths for a given hexcode
def get_image_paths(hexcode):
    image_paths = {}
    
    for brand in brands:
        brand_path = os.path.join(image_base_path, brand)
        if not os.path.exists(brand_path): # Skip if folder doesn't exist
            continue
            
        expected_filename = f"{hexcode}.pt"  # Adjust based on actual format
        
        if expected_filename in os.listdir(brand_path):
            image_paths[brand] = os.path.join(brand_path, expected_filename)

    return image_paths

# Expand dataframe with tqdm progress bar
expanded_rows = []
for _, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing Hexcodes"):
    hexcode = row["hexcode"]
    embedding = row["combined_embedding"]
    prompt = row["prompt"]
    skintone = row["skintone"]

    image_paths = get_image_paths(hexcode)  # Get list of image paths
    
    if image_paths:  # If images exist, create multiple rows
        for brand, path in image_paths.items():
            expanded_rows.append({"hexcode": hexcode, "prompt": prompt, "skintone":skintone, "combined_embedding": embedding, "image_path": path})
    else:
        # If no images exist, optionally add a row with NaN for image_path
        expanded_rows.append({"hexcode": hexcode, "prompt": prompt, "skintone":skintone, "combined_embedding": embedding, "image_path": None})

# Convert to DataFrame
expanded_df = pd.DataFrame(expanded_rows)

# Optional: Drop rows where no image is found
# expanded_df = expanded_df.dropna(subset=["image_path"]).reset_index(drop=True)



expanded_df.info()




# Try converting skintone column to numeric (if possible)
expanded_df['skintone'] = pd.to_numeric(expanded_df['skintone'], errors='raise')




output_file = '../data/processed_emoji_dataset.parquet'

# Check if the file exists and remove it
if os.path.exists(output_file):
    os.remove(output_file)

# Now save the DataFrame as Parquet
expanded_df.to_parquet(output_file, index=False)






