from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import nltk
from nltk.wsd import lesk
BASE = Path(__file__).resolve().parents[1]
app = Flask(__name__)
@app.get("/")
def home(): return send_from_directory(BASE / "frontend", "index.html")
@app.get("/<path:name>")
def assets(name): return send_from_directory(BASE / "frontend", name)
@app.post("/api/disambiguate")
def disambiguate():
    body = request.get_json(silent=True) or {}
    sentence = body.get("sentence", "").strip()
    target = body.get("word", "").strip().lower()
    if not sentence or not target: return jsonify({"error": "Enter a sentence and target word."}), 400
    try:
        tokens = nltk.word_tokenize(sentence)
        if target not in [t.lower() for t in tokens]:
            return jsonify({"error": "The target word must appear in the sentence."}), 400
        sense = lesk(tokens, target)
        if sense is None:
            return jsonify({"word": target, "synset": None, "definition": "No clear sense found. Add more context and try again.", "examples": [], "lemmas": []})
        return jsonify({"word": target, "synset": sense.name(), "definition": sense.definition(), "examples": sense.examples(), "lemmas": [l.name() for l in sense.lemmas()]})
    except LookupError:
        return jsonify({"error": "NLTK data unavailable. Run the download commands in README.md."}), 503
if __name__ == "__main__": app.run(debug=True, port=5005)
