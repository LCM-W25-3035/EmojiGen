# Importing necessary libraries

import re
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")




openmoji_df = pd.read_csv('../data/openmoji.csv')
emojipedia_df = pd.read_csv('../data/emojipedia.csv')
llm_df = pd.read_parquet('../data/llmemoji.parquet')

openmoji_df
openmoji_df.info()
openmoji_df.nunique()
openmoji_df.isnull().sum()
emojipedia_df
emojipedia_df.info()
emojipedia_df.nunique()
llm_df
llm_df.info()

llm_df[llm_df['unicode'] == 'U+1F947']

emojipedia_df[emojipedia_df['Codepoints Hex'] == 'U+0023,U+FE0F,U+20E3']

openmoji_df[openmoji_df['openmoji_tags'] == 'logo']

openmoji_df['openmoji_tags'].unique()

def normalize_unicode(unicode_str):
    # Remove existing "U+" prefixes
    cleaned = re.sub(r"U\+", "", unicode_str)

    # Replace separators (spaces, commas, hyphens) with a single space
    cleaned = re.sub(r"[\s,.-]+", " ", cleaned)

    # Add "U+" prefix to each code point
    normalized = " ".join(f"U+{code}" for code in cleaned.split())

    return normalized




# Merge the llm_emoji_df, openmoji_df, emojipedia_df based on unicode/hexcode
# The final df should have unicode, title (short descriotion, annotation), tags, group, subgroup, description)
llm_df.info()
llm_sm_df = llm_df[['character','unicode','tags','LLM description']]
llm_sm_df.rename(columns={'character': 'llm_emoji','tags': 'llm_emoji_tags','LLM description': 'llm_description'}, inplace=True)
llm_sm_df['unicode'] = llm_sm_df['unicode'].apply(normalize_unicode)

llm_sm_df




openmoji_df.info()
openmoji_sm_df = openmoji_df[['emoji','hexcode','group','subgroups','annotation','tags','openmoji_tags']]
openmoji_sm_df.rename(columns={'emoji':'openmoji_emoji','hexcode':'unicode', 'annotation':'openmoji_annotations'}, inplace=True)
openmoji_sm_df['unicode'] = openmoji_sm_df['unicode'].apply(normalize_unicode)

def merge_columns(col1, col2):
    if pd.isna(col1) and pd.isna(col2):  
        return float("NaN")  # Keep NaN if both are missing
    elif pd.isna(col1):  
        return col2  # If col1 is NaN, take col2
    elif pd.isna(col2):  
        return col1  # If col2 is NaN, take col1
    else:  
        return f"{col1},{col2}"  # If both have values, merge with a comma

openmoji_sm_df['openmoji_tags'] = openmoji_sm_df.apply(lambda row: merge_columns(row['tags'], row["openmoji_tags"]), axis=1)
openmoji_sm_df.drop(columns=["tags"], inplace=True)
openmoji_sm_df




emojipedia_df.info()
emojipedia_sm_df = emojipedia_df[['Emoji','Description','Codepoints Hex']]
emojipedia_sm_df.rename(columns={'Emoji':'emojipedia_emoji','Codepoints Hex':'unicode', 'Description':'emojipedia_description'}, inplace=True)
emojipedia_sm_df['unicode'] = emojipedia_sm_df['unicode'].apply(normalize_unicode)
emojipedia_sm_df




#Merging the datasets
final_df = llm_sm_df.merge(openmoji_sm_df, on='unicode', how='outer').merge(emojipedia_sm_df, on='unicode', how='outer')
final_df




def merge_columns(col1, col2):
    if pd.isna(col1) and pd.isna(col2):  
        return float("NaN")  # Keep NaN if both are missing
    elif pd.isna(col1):  
        return col2  # If col1 is NaN, take col2
    elif pd.isna(col2):  
        return col1  # If col2 is NaN, take col1
    else:  
        return f"{col1}"  # If both have values, replace with first




final_df['emoji'] = final_df.apply(lambda row: merge_columns(row['llm_emoji'], row["openmoji_emoji"]), axis=1)
final_df.drop(columns=["llm_emoji","openmoji_emoji"], inplace=True)
final_df['emoji'] = final_df.apply(lambda row: merge_columns(row['emoji'], row["emojipedia_emoji"]), axis=1)
final_df.drop(columns=["emojipedia_emoji"], inplace=True)
final_df
final_df.info()
final_df.isnull().sum()


from IPython.display import display, HTML

# Display the dataset in a scrollable box
display(HTML(final_df.to_html()))

