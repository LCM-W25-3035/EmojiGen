# 🚀 Prompt Summary
# First Prompt: Fetch individual emoji pages and extract their description and codepointsHex using requests and BeautifulSoup.
# Last Prompt: Improve error handling for unicode extraction and save data into CSV format.
# Model Used: ChatGPT (gpt-4-turbo)
# -----------------------------------------------

import os

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import csv
import requests
from bs4 import BeautifulSoup
import json
import time
import warnings
warnings.filterwarnings("ignore")

# File containing the saved emoji data
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

input_file = os.path.join(PROJECT_ROOT, "data", "emojipedia", "emojis_with_subgroups.csv")
output_file = os.path.join(PROJECT_ROOT, "data", "emojipedia", "emoji_details.csv")

# Base URL for Emojipedia
base_url = "https://emojipedia.org"

def extract_unicode(script_content):
    try:
        # Parse the JSON content from the script tag
        data = json.loads(script_content)
        # Navigate to the relevant section
        page_props = data.get("props", {}).get("pageProps", {})
        emoji_data = page_props.get("dehydratedState", {}).get("queries", [])
        for query in emoji_data:
            state_data = query.get("state", {}).get("data", [])
            if isinstance(state_data,dict) and 'codepointsHex' in state_data.keys():
                unicodes = ''
                
        
                for unicode in state_data['codepointsHex']:
                    
            
                    if unicodes == '':
                        unicodes = unicodes+unicode
                    else:
                        unicodes=unicodes+','+unicode
                    # unicodes.app
                return unicodes
            else:
                continue
    except json.JSONDecodeError:
        logging.info("Failed to parse JSON.")
    return None


# Function to fetch and parse a webpage
def fetch_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BeautifulSoup(response.content, "html.parser")
    else:
        logging.info(f"Failed to fetch {url}: {response.status_code}")
        return None

# Function to extract description and Unicode from the emoji page
def extract_emoji_details(emoji_url):
    soup = fetch_page(emoji_url)
    if not soup:
        return None, None, None

    # Extract the description
    description_section = soup.find("div", class_="HtmlContent_html-content-container__Ow2Bk")
    description = description_section.get_text(separator=" ", strip=True) if description_section else "No description available"
    description = description.replace("\n", " ")

    # Create the technical information URL
    unicode_url = emoji_url + "#technical"
    
    
    unicode_soup = fetch_page(unicode_url)
    if not unicode_soup:
        return description, "No Unicode available", "No Codepoints Hex available"

    # Extract `codepointsHex` from the JSON-like script
    script_tag = unicode_soup.find("script", text=lambda t: t and "codepointsHex" in t)
    codepoints_hex = "No Codepoints Hex available"


    if script_tag:
        
        
        codepoints_hex = extract_unicode(script_tag.string)

        # Parse the JSON-like content within the script tag
    return description, codepoints_hex

# Read the input CSV and process each emoji
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", newline="", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    fieldnames = ["Group","Subgroup", "Emoji", "Title", "DescribedBy", "URL", "Description", "Codepoints Hex"]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Process each row
    group=''
    for row in reader:
        if group != row["Group"]:
            group = row['Group']
            logging.info(group)
        emoji_url = base_url + row["URL"]

        # Extract description and Unicode details
        
        description, codepoints_hex = extract_emoji_details(emoji_url)

        # Add the details to the row and save to the output file
        row["Description"] = description
        row["Codepoints Hex"] = codepoints_hex
        writer.writerow(row)

        # Pause to avoid overloading the server
        time.sleep(0.5)

logging.info(f"Emoji details saved to {output_file}")
