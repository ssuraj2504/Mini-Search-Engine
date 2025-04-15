import os
from sklearn.feature_extraction.text import TfidfVectorizer

def load_documents(directory):
    documents = []
    filenames = []

    for fname in os.listdir(directory):
        if fname.endswith(".txt"):
            with open(os.path.join(directory, fname), 'r', encoding='utf-8') as f:
                documents.append(f.read())
                filenames.append(fname)

    return filenames, documents

def build_vectorizer(documents):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix
