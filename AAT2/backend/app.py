from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
BASE = Path(__file__).resolve().parents[1]
app = Flask(__name__)
DOCS = json.loads((BASE / "data" / "documents.json").read_text(encoding="utf-8"))
@app.get("/")
def home(): return send_from_directory(BASE / "frontend", "index.html")
@app.get("/<path:name>")
def assets(name): return send_from_directory(BASE / "frontend", name)
@app.post("/api/search")
def search():
    query = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not query: return jsonify({"error": "Type a question or search phrase."}), 400
    corpus = [d["text"] for d in DOCS]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:3]
    results = [{"title": DOCS[i]["title"], "text": DOCS[i]["text"], "score": round(float(score), 3)} for i, score in ranked if score > 0]
    return jsonify({"results": results, "answer": results[0]["text"] if results else "No close match found. Try different keywords."})
if __name__ == "__main__": app.run(debug=True, port=5002)
