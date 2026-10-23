from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
BASE = Path(__file__).resolve().parents[1]
app = Flask(__name__)
DOCS = [
"Neural networks learn patterns from data using layers, weights, activation functions, and gradient descent.",
"Machine learning models use training data to predict labels, values, and future outcomes.",
"Climate research studies rainfall, temperature, storms, and long-term environmental changes.",
"Weather forecasting combines observations, numerical models, and satellite measurements.",
"Natural language processing works with tokens, text classification, translation, and language models.",
"Search engines rank web pages using keywords, relevance signals, and document retrieval.",
"Deep learning uses neural networks for images, speech recognition, and language understanding.",
"Environmental monitoring tracks air quality, water levels, rainfall, and ecosystem health.",
"Topic modelling groups documents by recurring word patterns without requiring labels.",
"Information retrieval finds documents relevant to a user query from a collection."
]
@app.get("/")
def home(): return send_from_directory(BASE / "frontend", "index.html")
@app.get("/<path:name>")
def assets(name): return send_from_directory(BASE / "frontend", name)
@app.post("/api/topics")
def topics():
    body = request.get_json(silent=True) or {}
    try: n = int(body.get("n_topics", 3))
    except (ValueError, TypeError): n = 3
    n = max(2, min(5, n))
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(DOCS)
    model = LatentDirichletAllocation(n_components=n, random_state=21, max_iter=15).fit(matrix)
    words = vectorizer.get_feature_names_out()
    output = []
    for i, weights in enumerate(model.components_):
        top = weights.argsort()[-7:][::-1]
        output.append({"topic": i + 1, "words": [{"word": words[j], "weight": round(float(weights[j]), 2)} for j in top]})
    return jsonify({"topics": output, "documents": len(DOCS)})
if __name__ == "__main__": app.run(debug=True, port=5004)
