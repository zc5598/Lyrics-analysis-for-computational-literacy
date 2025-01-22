import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

path = "/Users/jcmac/Downloads/lyrics-analysis/cleaned-lyrics/70s"

documents = []
filenames = []
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        with open(os.path.join(path, filename), 'r') as file:
            documents.append(file.read())
            filenames.append(filename)

print(documents)

def display_topics(model, feature, topwords):
    for idx, topic in enumerate(model.components_):
        print(f"  Topic {idx + 1}:")
        print(", ".join([feature[i] for i in topic.argsort()[:-topwords - 1:-1]]))
    print("\n")


vectorizer = CountVectorizer(max_features=1000,stop_words ='english')  
X = vectorizer.fit_transform(documents) 

lda = LatentDirichletAllocation(n_components=1, random_state=42)
lda.fit(X)

print("Common theme by centuries:")
display_topics(lda, vectorizer.get_feature_names_out(), 10)
