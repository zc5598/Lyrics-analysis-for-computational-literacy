import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
    topics = {}
    for idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        topics[f"Topic {idx + 1}"] = top_words
        print(f"  Topic {idx + 1}: {', '.join(top_words)}")
    return topics

# visualize 
def visualize_themes(topics, file_name):
    for topic, words in topics.items():
        word_freq = {word: i+1 for i, word in enumerate(words)}  
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"{file_name} - {topic}")
        plt.show()

print("Analyzing themes for each song:")
for i, text in enumerate(documents):
    print(f"Song: {file_names[i]}")

    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform([text]) 
    
    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(X)
    

    topics = display_topics(lda, vectorizer.get_feature_names_out(), 5)
    visualize_themes(topics, file_names[i])
