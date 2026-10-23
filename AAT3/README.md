# AAT3 — Movie Knowledge Retrieval
A local RAG-style retrieval baseline: it finds relevant movie records and returns a summary grounded in those records. No hosted LLM or Neo4j instance is required.
```bash
python -m pip install -r backend/requirements.txt
cd backend
python app.py
```
Open http://127.0.0.1:5003. Sample data is in data/movies.json.
