import os
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

single_file = "/Users/jcmac/Downloads/lyrics-analysis/cleaned-lyrics/90s/i will always love you cleaned.txt"

with open(single_file, 'r') as file:
    lyrics = file.read()

result = classifier(lyrics, ["Love", "Freedom", "Dreams", "Faith", "Existential", "Loneliness"], multi_label=True)

print(os.path.basename(single_file))
for label, score in zip(result["labels"], result["scores"]):
    print(f"Theme: {label}, Confidence: {score:.2f}")
