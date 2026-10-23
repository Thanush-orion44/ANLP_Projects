# AAT5 — Word-Sense Disambiguation with Lesk
The Lesk heuristic compares context words with WordNet glosses to estimate the intended sense of an ambiguous word.
```bash
python -m pip install -r backend/requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"
cd backend
python app.py
```
Open http://127.0.0.1:5005. This is a heuristic prediction; overlap does not guarantee the intended meaning.
