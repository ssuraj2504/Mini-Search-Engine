from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re

app = Flask(__name__)
DATA_DIR = 'data'

# Load documents and filenames
filenames = []
documents = []

for fname in os.listdir(DATA_DIR):
    if fname.endswith(".txt"):
        with open(os.path.join(DATA_DIR, fname), 'r', encoding='utf-8') as file:
            documents.append(file.read())
            filenames.append(fname)

# Build the TF-IDF matrix
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Search Function using TF-IDF + Cosine Similarity
def search(query, top_n=10):
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[::-1][:top_n]

    results = []
    for idx in top_indices:
        score = cosine_sim[idx]
        if score == 0:
            continue

        content = documents[idx].lower()
        query_words = re.findall(r'\w+', query.lower())

        # Try to find a snippet with the first match
        for word in query_words:
            match = re.search(rf"(.{{0,40}}{word}.{{0,40}})", content, re.IGNORECASE)
            if match:
                snippet = match.group(1)
                break
        else:
            snippet = content[:100]

        # Highlight all query terms
        for word in query_words:
            snippet = re.sub(f"({word})", r"<mark>\1</mark>", snippet, flags=re.IGNORECASE)

        results.append({
            "filename": filenames[idx],
            "score": round(score, 2),
            "snippet": snippet + "..."
        })

    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_route():
    query = request.form.get('query')
    results = search(query)
    return render_template('results.html', query=query, results=results)

@app.route('/article/<filename>')
def view_article(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
        return render_template('article.html', filename=filename, content=content)
    else:
        return f"File '{filename}' not found.", 404

if __name__ == '__main__':
    app.run(debug=True)
