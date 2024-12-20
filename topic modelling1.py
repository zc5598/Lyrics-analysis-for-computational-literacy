import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

folder_path = "/Users/jcmac/Downloads/lyrics-analysis"

documents = []
file_names = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), 'r') as file:
            documents.append(file.read())
            file_names.append(filename)

print("Loaded Documents:")
print(documents)

def display_topics(model, feature_names, no_top_words):
    for idx, topic in enumerate(model.components_):
        print(f"  Topic {idx + 1}:")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    print("\n")

print("Analyzing themes for each song:")
for i, text in enumerate(documents):
    print(f"Song: {file_names[i]}")
 
    vectorizer = CountVectorizer(max_features=1000)  
    X = vectorizer.fit_transform([text]) 

    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(X)

    display_topics(lda, vectorizer.get_feature_names_out(), 5)
