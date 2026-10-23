# ANLP Projects — AAT 1 to AAT 5

Five independent Applied NLP demos with Flask APIs and browser interfaces.

| Folder | Experiment | Technique |
|---|---|---|
| AAT1 | POS tagging and named entity recognition | spaCy |
| AAT2 | Information retrieval and question answering | TF-IDF + cosine similarity |
| AAT3 | Movie knowledge retrieval (RAG-style baseline) | TF-IDF + grounded source records |
| AAT4 | Topic modelling | scikit-learn LDA |
| AAT5 | Word-sense disambiguation | NLTK WordNet + Lesk |

## Run a project
```bash
cd AAT2/backend
python -m pip install -r requirements.txt
python app.py
```
Open the local URL printed by Flask. Ports: AAT1 5001, AAT2 5002, AAT3 5003, AAT4 5004, AAT5 5005. Each app serves its own frontend.

AAT1 requires `python -m spacy download en_core_web_sm`. AAT5 may need WordNet downloads; see its README. AAT3 uses illustrative local sample data and does not require Neo4j or a hosted LLM.

These are educational demos, not production services. Implementations and interface themes are independently written.
