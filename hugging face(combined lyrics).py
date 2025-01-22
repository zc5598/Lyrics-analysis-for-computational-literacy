import os
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

path = "/Users/jcmac/Downloads/lyrics-analysis/cleaned-lyrics/90s"

alllyrics = " " 
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        with open(os.path.join(path, filename), 'r') as file:
            alllyrics += file.read() + "\n"

result = classifier(alllyrics, ["Love", "Freedom", "Dreams", "Faith", "Existential", "Loneliness"], multi_label=True)
print("Combined Analysis:")
for label, score in zip(result["labels"], result["scores"]):
    print(f"Theme: {label}, Confidence: {score:.2f}")