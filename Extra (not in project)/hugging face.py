import os
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# Deciding themes
themes = ["Love", "Freedom", "Dreams", "Faith","Existential","Loneliness"]

path = "/Users/jcmac/Downloads/lyrics-analysis/cleaned-lyrics/70s"

for filename in os.listdir(path):
    if filename.endswith(".txt"):
        file_path = os.path.join(path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            lyrics = file.read()  
            
            if lyrics.strip():  
                print(f"Analyzing: {filename}")
                result = classifier(lyrics[:500], themes, multi_label=True) 
                for label, score in zip(result["labels"], result["scores"]):
                    print(f"Theme: {label}, Confidence: {score:.2f}")
                print("-" * 50)
            else:
                print(f"Skipping empty file: {filename}")