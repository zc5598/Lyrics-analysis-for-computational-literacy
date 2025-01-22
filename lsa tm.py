import os
from gensim import corpora, models 
from gensim.parsing.preprocessing import preprocess_string, STOPWORDS 
from gensim.utils import simple_preprocess

path = "/Users/jcmac/Downloads/lyrics-analysis/cleaned-lyrics/70s"

documents = []
filenames = []
for filename in os.listdir(path):
    if filename.endswith(".txt"):
        with open(os.path.join(path, filename), 'r') as file:
            content = file.read()
            documents.append(content)
            filenames.append(filename)

print(f"Loaded {len(documents)} documents.")

def preprocess(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

preprocessed = [preprocess(doc) for doc in documents]

dictionary = corpora.Dictionary(preprocessed)
corpus = [dictionary.doc2bow(doc) for doc in preprocessed]

topics = 1  
lsamodel = models.LsiModel(corpus, id2word=dictionary, num_topics=topics)


print("Common Theme by century:")
topics = lsamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
