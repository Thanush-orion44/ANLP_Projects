from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
BASE = Path(__file__).resolve().parents[1]
app = Flask(__name__)
MOVIES = json.loads((BASE / "data" / "movies.json").read_text(encoding="utf-8"))
@app.get("/")
def home(): return send_from_directory(BASE / "frontend", "index.html")
@app.get("/<path:name>")
def assets(name): return send_from_directory(BASE / "frontend", name)
@app.post("/api/ask")
def ask():
    q = (request.get_json(silent=True) or {}).get("question", "").strip()
    if not q: return jsonify({"error": "Ask a movie question first."}), 400
    docs = [f"{m['title']} ({m['year']}). Genre: {m['genre']}. Plot: {m['plot']} Director: {m['director']}." for m in MOVIES]
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    x = v.fit_transform(docs + [q])
    scores = cosine_similarity(x[-1], x[:-1]).ravel()
    ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
    found = [{"title": MOVIES[i]["title"], "year": MOVIES[i]["year"], "genre": MOVIES[i]["genre"], "plot": MOVIES[i]["plot"], "director": MOVIES[i]["director"], "score": round(float(scores[i]), 3)} for i in ids if scores[i] > 0]
    answer = "Based on the retrieved records: " + " ".join(f"{m['title']} ({m['year']}) is a {m['genre']} film directed by {m['director']}. {m['plot']}" for m in found[:2]) if found else "I could not find a matching movie in the local sample. Try a title, genre, director, or plot keyword."
    return jsonify({"answer": answer, "sources": found})
if __name__ == "__main__": app.run(debug=True, port=5003)
