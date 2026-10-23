from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import spacy

BASE = Path(__file__).resolve().parents[1]
app = Flask(__name__)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None

@app.get("/")
def home():
    return send_from_directory(BASE / "frontend", "index.html")

@app.get("/<path:name>")
def assets(name):
    return send_from_directory(BASE / "frontend", name)

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model_loaded": nlp is not None})

@app.post("/api/analyze")
def analyze():
    if nlp is None:
        return jsonify({"error": "English model missing. Run: python -m spacy download en_core_web_sm"}), 503
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "Enter some text first."}), 400
    doc = nlp(text)
    return jsonify({
        "tokens": [{"text": t.text, "pos": t.pos_, "tag": t.tag_, "lemma": t.lemma_} for t in doc],
        "entities": [{"text": e.text, "label": e.label_, "description": spacy.explain(e.label_) or e.label_} for e in doc.ents]
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
