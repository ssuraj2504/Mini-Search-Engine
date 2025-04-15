from sklearn.metrics.pairwise import cosine_similarity
from .indexer import load_documents, build_vectorizer
import numpy as np

class SearchEngine:
    def __init__(self, data_dir):
        self.filenames, self.documents = load_documents(data_dir)
        self.vectorizer, self.tfidf_matrix = build_vectorizer(self.documents)

    def search(self, query, top_n=10):
        query_vec = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(cosine_sim)[::-1][:top_n]

        results = []
        for idx in top_indices:
            if cosine_sim[idx] > 0:
                snippet = self.documents[idx][:200].replace('\n', ' ') + "..."
                results.append({
                    "filename": self.filenames[idx],
                    "score": round(cosine_sim[idx], 2),
                    "snippet": snippet
                })

        return results
