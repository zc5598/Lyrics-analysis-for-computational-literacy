import os
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

path = "/Users/jcmac/Downloads/lyrics-analysis/cleaned-lyrics"
alllyrics = []
titles = []

for decade in os.listdir(path):
    dpath = os.path.join(path, decade)
    if os.path.isdir(dpath):
        for filename in os.listdir(dpath):
            if filename.endswith(".txt"):
                with open(os.path.join(dpath, filename), 'r', encoding='utf-8') as file:
                    alllyrics.append(file.read())
                    titles.append(filename)

vectorizer = CountVectorizer(max_df=0.95, min_df=2) 
x = vectorizer.fit_transform(alllyrics)
topics = 6
ldamodel = LatentDirichletAllocation(n_components=topics, random_state=42)
ldamodel.fit(x)

def display(model, features, number_words=1):
    for topic_idx, topic in enumerate(model.components_):
        top_words = [features[i] for i in topic.argsort()[-number_words:]]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

print("Interpreted topics:")
display(ldamodel, vectorizer.get_feature_names_out())

distribution = ldamodel.transform(x)
distribution = normalize(distribution, norm='l1', axis=1)  

topicsdf = pd.DataFrame(distribution, columns=[f'Topic {i+1}' for i in range(topics)], index=titles)

print(topicsdf)

