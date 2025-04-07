import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
label_text='Number of Emojis'
file_path = '../data/openmoji.csv' 
df = pd.read_csv(file_path)

df.info()

df.head()


group_counts = df["group"].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(
    x=group_counts.values, 
    y=group_counts.index, 
    hue=group_counts.index,
    palette="viridis", 
    legend=False
)
plt.xlabel(label_text)
plt.ylabel("Emoji Group")
plt.title("Distribution of Emojis Across Different Groups")
plt.show()


subgroup_counts = df["subgroups"].value_counts().head(15)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=subgroup_counts.values, 
    y=subgroup_counts.index, 
    hue=subgroup_counts.index,  
    palette="magma", 
    legend=False 
)
plt.xlabel("Number of Emojis")
plt.ylabel("Emoji Subgroup")
plt.title("Top 15 Emoji Subgroups by Count")
plt.show()


from collections import Counter
from wordcloud import WordCloud

text_data = " ".join(df["annotation"].dropna().astype(str)) + " " + " ".join(df["tags"].dropna().astype(str))
word_freq = Counter(text_data.lower().split())
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate_from_frequencies(word_freq)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Common Words in Emoji Annotations & Tags")
plt.show()



skin_tone_base_counts = df["skintone_base_hexcode"].dropna().value_counts().head(15)  # Top 15

plt.figure(figsize=(12, 6))
sns.barplot(
    x=skin_tone_base_counts.values, 
    y=skin_tone_base_counts.index, 
    hue=skin_tone_base_counts.index, 
    palette="coolwarm", 
    legend=False
)
plt.xlabel(label_text)
plt.ylabel("Skin Tone Base Hexcode")
plt.title("Top 15 Skin Tone Base Hexcodes in Emojis")
plt.show()

skin_tone_combo_counts = df["skintone_combination"].dropna().value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(
    x=skin_tone_combo_counts.values, 
    y=skin_tone_combo_counts.index, 
    hue=skin_tone_combo_counts.index,
    palette="magma", 
    legend=False  
)
plt.xlabel(label_text)
plt.ylabel("Skin Tone Combinations")
plt.title("Skin Tone Combinations in Emojis")
plt.xticks(rotation=45)
plt.show()

